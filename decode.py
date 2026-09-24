#!/usr/bin/env python3
"""
decode.py

Experimental pure-standard-library Meshtastic decoder for SDRconnect IQ WAV files.

No third-party Python packages are required.
"""

import argparse
import cmath
import math
import os
import struct
import sys

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SF = 11
DEFAULT_BW = 250_000
DEFAULT_PREAMBLE = 16

# Meshtastic default channel key ("AQ==" expanded)
DEFAULT_PSK = bytes.fromhex("d4f1bb3a20290759f0bcffabcf4e6901")

# ---------------------------------------------------------------------------
# WAV / RF64 reader
# ---------------------------------------------------------------------------

def read_u32le(b, off=0):
    return struct.unpack_from("<I", b, off)[0]

def read_u64le(b, off=0):
    return struct.unpack_from("<Q", b, off)[0]

def parse_iq_wav(path):
    """
    Return (sample_rate, iq_complex_list).

    Supports ordinary RIFF/WAVE and RF64/WAVE with PCM 16-bit stereo IQ.
    RF64 sizes are resolved from ds64 where needed.
    """
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) != 12:
            raise ValueError("File too short for WAV/RF64 header")

        riff_id = head[0:4]
        wave_id = head[8:12]
        if riff_id not in (b"RIFF", b"RF64") or wave_id != b"WAVE":
            raise ValueError("Not a RIFF/RF64 WAVE file")

        rf64 = riff_id == b"RF64"
        ds64_data_size = None
        fmt = None
        data_pos = None
        data_size = None

        while True:
            chunk_hdr = f.read(8)
            if len(chunk_hdr) < 8:
                break

            cid = chunk_hdr[:4]
            csize = read_u32le(chunk_hdr, 4)
            cpos = f.tell()

            if cid == b"ds64":
                raw = f.read(csize)
                if len(raw) >= 24:
                    # riffSize, dataSize, sampleCount
                    ds64_data_size = read_u64le(raw, 8)

            elif cid == b"fmt ":
                raw = f.read(csize)
                if len(raw) < 16:
                    raise ValueError("Invalid fmt chunk")
                audio_fmt, channels, sample_rate, byte_rate, block_align, bits = \
                    struct.unpack_from("<HHIIHH", raw, 0)
                fmt = (audio_fmt, channels, sample_rate, block_align, bits)

            elif cid == b"data":
                data_pos = f.tell()
                if rf64 and csize == 0xFFFFFFFF and ds64_data_size is not None:
                    data_size = ds64_data_size
                else:
                    data_size = csize
                f.seek(data_size, os.SEEK_CUR)

            else:
                f.seek(csize, os.SEEK_CUR)

            if csize & 1:
                f.seek(1, os.SEEK_CUR)

            if fmt and data_pos is not None:
                break

        if fmt is None or data_pos is None:
            raise ValueError("Missing fmt or data chunk")

        audio_fmt, channels, sample_rate, block_align, bits = fmt

        # PCM = 1. WAVE_FORMAT_EXTENSIBLE = 0xFFFE; many SDR files use it,
        # but this alpha version still expects integer PCM samples.
        if audio_fmt not in (1, 0xFFFE):
            raise ValueError(f"Unsupported WAV format code {audio_fmt}")
        if channels < 2:
            raise ValueError(f"Need at least 2 channels for I/Q; got {channels}")
        if bits != 16:
            raise ValueError(f"Initial decoder supports 16-bit PCM only; got {bits}-bit")
        if block_align < channels * 2:
            raise ValueError("Unexpected block alignment")

        f.seek(data_pos)
        raw = f.read(data_size)

    frame_bytes = block_align
    frames = len(raw) // frame_bytes
    iq = [0j] * frames
    scale = 1.0 / 32768.0

    # First two channels are taken as I and Q.
    for n in range(frames):
        o = n * frame_bytes
        i = struct.unpack_from("<h", raw, o)[0]
        q = struct.unpack_from("<h", raw, o + 2)[0]
        iq[n] = complex(i * scale, q * scale)

    return sample_rate, iq

# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------

def fft_inplace(a):
    """Radix-2 complex FFT, in place."""
    n = len(a)
    if n == 0 or (n & (n - 1)):
        raise ValueError("FFT length must be a power of two")

    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        wlen = cmath.exp(-2j * math.pi / length)
        half = length >> 1
        for i in range(0, n, length):
            w = 1 + 0j
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * w
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= wlen
        length <<= 1

def ideal_upchirp(n):
    """
    Generate one nominal LoRa upchirp at Fs=BW, N=2^SF samples/symbol.
    """
    out = [0j] * n
    for k in range(n):
        # Baseband sweep from -BW/2 to +BW/2, normalized to Fs=BW.
        phase = 2.0 * math.pi * ((k * k) / (2.0 * n) - k / 2.0)
        out[k] = cmath.exp(1j * phase)
    return out

def decimate_integer(samples, factor):
    if factor == 1:
        return samples
    # Boxcar decimator. Crude but dependency-free and adequate for a first pass
    # when the desired LoRa channel is already centered in the recording.
    out = []
    inv = 1.0 / factor
    for i in range(0, len(samples) - factor + 1, factor):
        s = 0j
        for x in samples[i:i+factor]:
            s += x
        out.append(s * inv)
    return out

def freq_shift(samples, fs, hz):
    if abs(hz) < 1e-9:
        return samples
    step = -2.0 * math.pi * hz / fs
    w = 1 + 0j
    dw = cmath.exp(1j * step)
    out = [0j] * len(samples)
    for i, x in enumerate(samples):
        out[i] = x * w
        w *= dw
        if (i & 0xFFFF) == 0:
            # Control accumulated floating-point error.
            mag = abs(w)
            if mag:
                w /= mag
    return out

def dechirp_peak(samples, pos, upchirp):
    n = len(upchirp)
    if pos < 0 or pos + n > len(samples):
        return None
    v = [samples[pos + k] * upchirp[k].conjugate() for k in range(n)]
    fft_inplace(v)
    powers = [z.real*z.real + z.imag*z.imag for z in v]
    idx = max(range(n), key=powers.__getitem__)
    p = powers[idx]
    avg = sum(powers) / n + 1e-30
    return idx, p / avg

def circular_distance(a, b, n):
    d = abs(a - b)
    return min(d, n - d)

def chirp_coherence(samples, pos, ref):
    """Return normalized adjacent-sample coherence after dechirping."""
    n = len(ref)
    if pos < 0 or pos + n > len(samples):
        return 0.0
    prev = samples[pos] * ref[0].conjugate()
    acc = 0j
    power = 0.0
    for k in range(1, n):
        cur = samples[pos+k] * ref[k].conjugate()
        acc += cur * prev.conjugate()
        power += abs(cur) * abs(prev)
        prev = cur
    return abs(acc) / (power + 1e-30)

def find_preamble(samples, upchirp, min_run=7, ratio_threshold=10.0):
    """
    Find a LoRa preamble while automatically trying normal and spectrally
    inverted chirps. Returns (symbol_boundary, cfo_bin, inverted).
    """
    n = len(upchirp)
    refs = ((False, upchirp), (True, [z.conjugate() for z in upchirp]))
    hop = max(1, n // 4)
    coh_threshold = 0.72 if ratio_threshold >= 8 else 0.62

    for inverted, ref in refs:
        run = []
        for pos in range(0, len(samples) - n, hop):
            coh = chirp_coherence(samples, pos, ref)
            if coh >= coh_threshold:
                run.append((pos, coh))
            else:
                run = []

            if len(run) >= max(12, min_run * 3):
                approx = run[0][0]
                best = None
                search_start = max(0, approx - n)
                search_end = min(len(samples) - n * min_run, approx + n)
                step = max(1, n // 128)

                for cand in range(search_start, search_end + 1, step):
                    bins = []
                    score = 0.0
                    ok = True
                    for s in range(min_run):
                        got = dechirp_peak(samples, cand + s*n, ref)
                        if got is None:
                            ok = False
                            break
                        bb, rr = got
                        bins.append(bb)
                        score += math.log(max(rr, 1e-9))
                    if not ok:
                        continue
                    spread = sum(circular_distance(x, bins[0], n) for x in bins[1:])
                    metric = score - spread
                    if best is None or metric > best[0]:
                        best = (metric, cand, bins[0])

                if best:
                    return best[1], best[2], inverted

    return None, None, None

def demod_symbol(samples, pos, ref, cfo_bin=0, inverted=False):
    got = dechirp_peak(samples, pos, ref)
    if got is None:
        raise EOFError
    idx, ratio = got
    n = len(ref)
    sym = (cfo_bin - idx) % n if inverted else (idx - cfo_bin) % n
    return sym, ratio

# ---------------------------------------------------------------------------
# LoRa PHY decoding
# ---------------------------------------------------------------------------

def binary_to_gray(x):
    return x ^ (x >> 1)

def rotl(x, r, width):
    r %= width
    mask = (1 << width) - 1
    return ((x << r) | (x >> (width-r if r else width))) & mask if r else x & mask

def diagonal_deinterleave(symbols, symbol_bits, parity_bits):
    """
    Inverse diagonal interleaver. Consumes groups of (4+parity_bits) symbols
    and produces symbol_bits codewords per group.
    """
    cols = 4 + parity_bits
    out = []
    for base in range(0, len(symbols) - cols + 1, cols):
        group = symbols[base:base+cols]
        codewords = [0] * symbol_bits
        for k in range(cols):
            word = group[k]
            for m in range(symbol_bits):
                i = (m + k) % symbol_bits
                bit = (word >> m) & 1
                codewords[i] |= bit << k
        out.extend(codewords)
    return out

def enc54(x):
    x &= 0xF
    p = x ^ (x >> 2)
    p ^= p >> 1
    return x | ((p << 4) & 0x10)

def enc64(x):
    x &= 0xF
    a = x ^ (x >> 1) ^ (x >> 2)
    b = a ^ x ^ (x >> 3)
    return x | ((a & 1) << 4) | ((b & 1) << 5)

def enc74(x):
    x &= 0xF
    d0 = (x >> 0) & 1
    d1 = (x >> 1) & 1
    d2 = (x >> 2) & 1
    d3 = (x >> 3) & 1
    return (
        x |
        ((d0 ^ d1 ^ d2) << 4) |
        ((d1 ^ d2 ^ d3) << 5) |
        ((d0 ^ d1 ^ d3) << 6)
    )

def enc84(x):
    x &= 0xF
    d0 = (x >> 0) & 1
    d1 = (x >> 1) & 1
    d2 = (x >> 2) & 1
    d3 = (x >> 3) & 1
    return (
        x |
        ((d0 ^ d1 ^ d2) << 4) |
        ((d1 ^ d2 ^ d3) << 5) |
        ((d0 ^ d1 ^ d3) << 6) |
        ((d0 ^ d2 ^ d3) << 7)
    )

def hamming_distance(a, b):
    return (a ^ b).bit_count()

def decode_codeword(cw, cr):
    """
    cr is 1..4 for 4/5 .. 4/8.
    Nearest-codeword decoding.
    """
    enc = {1: enc54, 2: enc64, 3: enc74, 4: enc84}[cr]
    mask = (1 << (4 + cr)) - 1
    cw &= mask
    best_n = 0
    best_d = 99
    for n in range(16):
        d = hamming_distance(cw, enc(n))
        if d < best_d:
            best_n = n
            best_d = d
    return best_n, best_d

def whitening_bytes(length):
    s = 0xFF
    out = bytearray()
    for _ in range(length):
        out.append(s)
        fb = ((s >> 7) ^ (s >> 5) ^ (s >> 4) ^ (s >> 3)) & 1
        s = ((s << 1) | fb) & 0xFF
    return bytes(out)

def pack_nibbles_low_first(nibbles):
    out = bytearray()
    for i in range(0, len(nibbles) - 1, 2):
        out.append((nibbles[i] & 0xF) | ((nibbles[i+1] & 0xF) << 4))
    return bytes(out)

def decode_lora_frame(symbols, sf):
    """
    Decode an explicit-header LoRa frame from already-demodulated symbol indices.

    Returns payload bytes and PHY metadata.
    """
    if len(symbols) < 8:
        raise ValueError("Not enough symbols for explicit LoRa header")

    # First 8 symbols use SF-2 effective bits and CR=4/8.
    hb = sf - 2
    hsyms = []
    shift = sf - hb
    for s in symbols[:8]:
        # LoRa adds +1 to every transmitted symbol after Gray mapping.
        # Undo that first; explicit-header symbols then have two padding LSBs.
        s = (s - 1) & ((1 << sf) - 1)
        s2 = s >> shift
        hsyms.append(binary_to_gray(s2) & ((1 << hb) - 1))

    hcws = diagonal_deinterleave(hsyms, hb, 4)
    if len(hcws) < 5:
        raise ValueError("Could not deinterleave PHY header")

    hn = []
    hdists = []
    for cw in hcws[:5]:
        n, d = decode_codeword(cw, 4)
        hn.append(n)
        hdists.append(d)

    payload_len = ((hn[0] & 0xF) << 4) | (hn[1] & 0xF)
    opts = hn[2] & 0xF
    cr = (opts >> 1) & 0x7
    has_crc = bool(opts & 1)

    if cr < 1 or cr > 4:
        raise ValueError(f"Invalid coding rate in LoRa header: {cr}")
    if payload_len < 1 or payload_len > 255:
        raise ValueError(f"Invalid payload length: {payload_len}")

    # Payload codewords that shared the first header interleave block.
    first_payload_cws = hcws[5:]

    # Remaining symbols use full SF at payload CR.
    psyms_raw = symbols[8:]
    cols = 4 + cr
    usable = (len(psyms_raw) // cols) * cols
    psyms_raw = psyms_raw[:usable]
    # Undo LoRa's +1 transmitted-symbol offset before Gray decoding.
    mask = (1 << sf) - 1
    psyms = [binary_to_gray((s - 1) & mask) & mask for s in psyms_raw]
    pcws = first_payload_cws + diagonal_deinterleave(psyms, sf, cr)

    needed_nibbles = payload_len * 2
    nibbles = []
    dists = []

    # The codewords that share the header block are protected with CR 4/8.
    for cw in first_payload_cws:
        n, d = decode_codeword(cw, 4)
        nibbles.append(n)
        dists.append(d)
        if len(nibbles) >= needed_nibbles:
            break

    if len(nibbles) < needed_nibbles:
        rest = pcws[len(first_payload_cws):]
        for cw in rest:
            n, d = decode_codeword(cw, cr)
            nibbles.append(n)
            dists.append(d)
            if len(nibbles) >= needed_nibbles:
                break

    if len(nibbles) < needed_nibbles:
        raise ValueError(
            f"Frame truncated: need {needed_nibbles} payload nibbles, got {len(nibbles)}"
        )

    payload = bytearray(pack_nibbles_low_first(nibbles[:needed_nibbles]))

    # Standard LoRa whitening over payload bytes.
    w = whitening_bytes(len(payload))
    for i in range(len(payload)):
        payload[i] ^= w[i]

    meta = {
        "payload_len": payload_len,
        "cr": cr,
        "has_crc": has_crc,
        "header_fec_distance": sum(hdists),
        "payload_fec_distance": sum(dists),
    }
    return bytes(payload), meta

# ---------------------------------------------------------------------------
# AES-128 / AES-256, standard-library only
# ---------------------------------------------------------------------------

SBOX = (
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
)

RCON = (0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36)

def aes_expand_key(key):
    if len(key) not in (16, 32):
        raise ValueError("AES key must be 16 or 32 bytes")
    nk = len(key) // 4
    nr = 10 if nk == 4 else 14
    words = [list(key[i:i+4]) for i in range(0, len(key), 4)]
    i = nk
    while len(words) < 4 * (nr + 1):
        temp = words[-1][:]
        if i % nk == 0:
            temp = temp[1:] + temp[:1]
            temp = [SBOX[x] for x in temp]
            temp[0] ^= RCON[i // nk]
        elif nk > 6 and i % nk == 4:
            temp = [SBOX[x] for x in temp]
        words.append([a ^ b for a, b in zip(words[i-nk], temp)])
        i += 1
    rounds = []
    for r in range(nr + 1):
        rounds.append(bytes(sum(words[4*r:4*r+4], [])))
    return rounds

def xtime(a):
    return ((a << 1) ^ 0x1B) & 0xFF if (a & 0x80) else (a << 1) & 0xFF

def aes_add_round_key(s, rk):
    for i in range(16):
        s[i] ^= rk[i]

def aes_sub_bytes(s):
    for i in range(16):
        s[i] = SBOX[s[i]]

def aes_shift_rows(s):
    # state is column-major
    t = s[:]
    s[0], s[4], s[8], s[12] = t[0], t[4], t[8], t[12]
    s[1], s[5], s[9], s[13] = t[5], t[9], t[13], t[1]
    s[2], s[6], s[10], s[14] = t[10], t[14], t[2], t[6]
    s[3], s[7], s[11], s[15] = t[15], t[3], t[7], t[11]

def aes_mix_columns(s):
    for c in range(4):
        i = 4*c
        a0,a1,a2,a3 = s[i:i+4]
        t = a0 ^ a1 ^ a2 ^ a3
        u = a0
        s[i]   ^= t ^ xtime(a0 ^ a1)
        s[i+1] ^= t ^ xtime(a1 ^ a2)
        s[i+2] ^= t ^ xtime(a2 ^ a3)
        s[i+3] ^= t ^ xtime(a3 ^ u)

def aes_encrypt_block(block, key):
    if len(block) != 16:
        raise ValueError("AES block must be 16 bytes")
    rks = aes_expand_key(key)
    nr = len(rks) - 1
    s = list(block)
    aes_add_round_key(s, rks[0])
    for r in range(1, nr):
        aes_sub_bytes(s)
        aes_shift_rows(s)
        aes_mix_columns(s)
        aes_add_round_key(s, rks[r])
    aes_sub_bytes(s)
    aes_shift_rows(s)
    aes_add_round_key(s, rks[nr])
    return bytes(s)

def aes_ctr_meshtastic(data, key, packet_id, from_node):
    """
    Meshtastic channel AES-CTR nonce:
       packet_id as uint64 LE
       from_node as uint32 LE
       block counter as uint32, starting at 0

    Firmware increments the final 32-bit counter in big-endian CTR fashion.
    """
    prefix = struct.pack("<Q", packet_id) + struct.pack("<I", from_node)
    out = bytearray()
    for block_no, pos in enumerate(range(0, len(data), 16)):
        ctr = prefix + struct.pack(">I", block_no)
        ks = aes_encrypt_block(ctr, key)
        chunk = data[pos:pos+16]
        out.extend(a ^ b for a, b in zip(chunk, ks))
    return bytes(out)

# ---------------------------------------------------------------------------
# Meshtastic parsing
# ---------------------------------------------------------------------------

def read_varint(buf, pos):
    val = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7
        if shift > 70:
            raise ValueError("Invalid protobuf varint")
    raise ValueError("Truncated protobuf varint")

def parse_protobuf_fields(buf):
    """
    Minimal generic protobuf parser sufficient for Meshtastic Data.
    Returns list of (field_number, wire_type, value).
    """
    fields = []
    p = 0
    while p < len(buf):
        key, p = read_varint(buf, p)
        field = key >> 3
        wt = key & 7

        if field == 0:
            raise ValueError("Invalid protobuf field 0")

        if wt == 0:
            v, p = read_varint(buf, p)
        elif wt == 1:
            if p + 8 > len(buf): raise ValueError("Truncated fixed64")
            v = buf[p:p+8]
            p += 8
        elif wt == 2:
            n, p = read_varint(buf, p)
            if p + n > len(buf): raise ValueError("Truncated length-delimited field")
            v = buf[p:p+n]
            p += n
        elif wt == 5:
            if p + 4 > len(buf): raise ValueError("Truncated fixed32")
            v = buf[p:p+4]
            p += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wt}")
        fields.append((field, wt, v))
    return fields

def parse_meshtastic_radio_payload(payload, psk):
    """
    Parse 16-byte Meshtastic radio header followed by encrypted Data protobuf.

    Header layout used by current Meshtastic radio transport:
      uint32 to
      uint32 from
      uint32 id
      uint8  flags
      uint8  channel
      uint8  next_hop
      uint8  relay_node
    """
    if len(payload) < 17:
        raise ValueError("Meshtastic radio payload too short")

    to_node, from_node, packet_id = struct.unpack_from("<III", payload, 0)
    flags = payload[12]
    channel_hash = payload[13]
    next_hop = payload[14]
    relay_node = payload[15]
    enc = payload[16:]

    clear = aes_ctr_meshtastic(enc, psk, packet_id, from_node)

    fields = parse_protobuf_fields(clear)
    portnum = None
    app_payload = None
    reply_id = None

    for field, wt, value in fields:
        if field == 1 and wt == 0:
            portnum = value
        elif field == 2 and wt == 2:
            app_payload = value
        elif field == 7 and wt == 5:
            reply_id = struct.unpack("<I", value)[0]

    return {
        "to": to_node,
        "from": from_node,
        "id": packet_id,
        "flags": flags,
        "channel_hash": channel_hash,
        "next_hop": next_hop,
        "relay_node": relay_node,
        "portnum": portnum,
        "app_payload": app_payload,
        "reply_id": reply_id,
        "clear_data": clear,
    }

def node_id(n):
    return f"!{n:08x}"

# ---------------------------------------------------------------------------
# Main decode flow
# ---------------------------------------------------------------------------

def decode_file(args):
    print(f"[+] Reading {args.wav}")
    fs, iq = parse_iq_wav(args.wav)
    print(f"[+] WAV sample rate: {fs} Hz")
    print(f"[+] IQ frames: {len(iq)}")

    if args.offset:
        print(f"[+] Frequency shifting by {args.offset:+.1f} Hz")
        iq = freq_shift(iq, fs, args.offset)

    if fs % args.bw != 0:
        raise ValueError(
            f"Sample rate {fs} is not an integer multiple of BW {args.bw}. "
            "This alpha build only supports integer decimation."
        )

    decim = fs // args.bw
    if decim < 1:
        raise ValueError("WAV sample rate must be >= LoRa bandwidth")

    print(f"[+] Decimating by {decim} -> {fs//decim} samples/s")
    iq = decimate_integer(iq, decim)

    n = 1 << args.sf
    up = ideal_upchirp(n)

    found = 0
    decoded = 0
    seen_packets = set()
    search_base = 0

    while search_base + n * (args.preamble + 5) < len(iq):
        sub = iq[search_base:]
        preamble_pos, cfo_bin, inverted = find_preamble(
            sub, up,
            min_run=min(8, args.preamble),
            ratio_threshold=args.threshold
        )

        if preamble_pos is None:
            break

        preamble_pos += search_base
        found += 1
        ref = [z.conjugate() for z in up] if inverted else up

        # The complete LoRa preamble consumes Npreamble + 4.25 symbols.
        data_pos = preamble_pos + int(round((args.preamble + 4.25) * n))

        if args.all_candidates or args.verbose:
            print()
            print(f"[+] Candidate frame #{found}")
            print(f"    preamble sample: {preamble_pos}")
            print(f"    CFO bin:         {cfo_bin}")
            print(f"    spectrum:        {'inverted' if inverted else 'normal'}")
            print(f"    data sample:     {data_pos}")

        # Collect enough raw symbols for max-sized frame; PHY header tells us
        # how much data is actually required.
        syms = []
        ratios = []
        pos = data_pos
        for _ in range(args.max_symbols):
            if pos + n > len(iq):
                break
            s, r = demod_symbol(iq, pos, ref, cfo_bin, inverted)
            syms.append(s)
            ratios.append(r)
            pos += n

        try:
            lora_payload, meta = decode_lora_frame(syms, args.sf)
            if args.all_candidates or args.verbose:
                print(
                    f"    LoRa payload:    {len(lora_payload)} bytes, "
                    f"CR=4/{meta['cr']+4}, CRC={'on' if meta['has_crc'] else 'off'}"
                )

            if args.dump_hex and (args.all_candidates or args.verbose):
                print(f"    raw:             {lora_payload.hex()}")

            pkt = parse_meshtastic_radio_payload(lora_payload, args.psk)

            packet_key = (pkt["from"], pkt["id"])
            duplicate = packet_key in seen_packets
            seen_packets.add(packet_key)

            if args.all_candidates or args.verbose:
                print(f"    from:            {node_id(pkt['from'])}")
                print(f"    to:              {node_id(pkt['to']) if pkt['to'] != 0xffffffff else 'broadcast'}")
                print(f"    packet id:       0x{pkt['id']:08x}")
                print(f"    channel hash:    0x{pkt['channel_hash']:02x}")
                print(f"    portnum:         {pkt['portnum']}")
                if duplicate:
                    print("    duplicate:       yes")

            if pkt["portnum"] == 1 and pkt["app_payload"] is not None:
                try:
                    text = pkt["app_payload"].decode("utf-8")
                except UnicodeDecodeError:
                    text = pkt["app_payload"].decode("utf-8", "replace")

                if not duplicate or args.show_duplicates:
                    decoded += 1
                    if args.all_candidates or args.verbose:
                        print(f"    MESSAGE:         {text}")
                    else:
                        dest = "broadcast" if pkt["to"] == 0xffffffff else node_id(pkt["to"])
                        print(f"{node_id(pkt['from'])} -> {dest}: {text}")

            elif pkt["app_payload"] is not None and (args.all_candidates or args.verbose):
                print(f"    app payload:     {pkt['app_payload'].hex()}")
            elif args.all_candidates or args.verbose:
                print("    decoded Data protobuf did not contain field 2")

        except Exception as e:
            if args.all_candidates or args.verbose:
                print(f"    [!] Decode failed: {e}")
                if args.verbose:
                    print("    First symbols:", syms[:24])

        # Move far enough forward not to rediscover the same preamble.
        search_base = preamble_pos + n * (args.preamble + 5)

    if found == 0:
        print("[!] No LoRa preamble candidates found.")
        print("    Check: center frequency, --offset, SF/BW, IQ channel order, and gain.")
    elif decoded == 0:
        print("[!] No Meshtastic text messages decoded.")
        if not (args.all_candidates or args.verbose):
            print("    Re-run with -v or --all-candidates for diagnostics.")
    elif args.all_candidates or args.verbose:
        print()
        print(f"[+] Examined {found} candidate frame(s); emitted {decoded} message(s).")

def parse_psk(s):
    s = s.strip()
    if s.lower() in ("default", "longfast"):
        return DEFAULT_PSK
    if s.lower() in ("none", "clear", "cleartext"):
        return bytes(16)
    s2 = s.replace(":", "").replace(" ", "")
    b = bytes.fromhex(s2)
    if len(b) not in (16, 32):
        raise argparse.ArgumentTypeError("PSK must be 16 or 32 bytes in hex, or 'default'")
    return b

def main():
    ap = argparse.ArgumentParser(
        description="Experimental pure-Python Meshtastic decoder for SDRconnect IQ WAV recordings."
    )
    ap.add_argument("wav", help="SDRconnect IQ WAV/RF64 file")
    ap.add_argument("--sf", type=int, default=DEFAULT_SF, help="LoRa spreading factor (default: 11)")
    ap.add_argument("--bw", type=int, default=DEFAULT_BW, help="LoRa bandwidth in Hz (default: 250000)")
    ap.add_argument("--preamble", type=int, default=DEFAULT_PREAMBLE, help="LoRa preamble symbols (default: 16)")
    ap.add_argument(
        "--psk", type=parse_psk, default=DEFAULT_PSK,
        help="Meshtastic channel PSK: 'default' or 16/32-byte hex (default: default)"
    )
    ap.add_argument(
        "--offset", type=float, default=0.0,
        help="Frequency offset in Hz: positive means shift a signal above WAV center down to baseband"
    )
    ap.add_argument(
        "--threshold", type=float, default=10.0,
        help="Preamble FFT peak/average threshold (default: 10)"
    )
    ap.add_argument(
        "--max-symbols", type=int, default=400,
        help="Maximum LoRa payload symbols to inspect per candidate (default: 400)"
    )
    ap.add_argument("--dump-hex", action="store_true", help="Print recovered LoRa payload bytes")
    ap.add_argument("--all-candidates", action="store_true",
                    help="Show false/failed LoRa candidates as well as valid packets")
    ap.add_argument("--show-duplicates", action="store_true",
                    help="Print repeated copies of the same Meshtastic packet ID")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not (7 <= args.sf <= 12):
        ap.error("This alpha build supports SF7..SF12")
    if args.bw <= 0:
        ap.error("Bandwidth must be positive")

    try:
        decode_file(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
