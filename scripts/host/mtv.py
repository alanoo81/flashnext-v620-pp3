#!/usr/bin/env python3
"""Lance memtest_vulkan dans un pseudo-terminal, choisit la carte <index> (1-based, comme dans sa liste),
laisse tourner <secondes>, écrit la sortie dans <log>, puis tue le groupe de processus.
Usage: mtv.py <index> <secondes> <log>"""
import os
import pty
import select
import signal
import sys
import time

idx, dur, log = sys.argv[1], float(sys.argv[2]), sys.argv[3]
pid, fd = pty.fork()
if pid == 0:
    os.chdir("/root/memtest_vulkan")
    os.execv("/root/memtest_vulkan/memtest_vulkan", ["memtest_vulkan"])
sent = False
buf = b""
t0 = time.time()
with open(log, "wb") as out:
    while time.time() - t0 < dur:
        r, _, _ = select.select([fd], [], [], 0.5)
        if fd in r:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            out.write(data)
            out.flush()
            buf += data
            # la liste des périphériques est affichée : envoyer le choix une seule fois
            if not sent and (b"llvmpipe" in buf or b"Override" in buf or b"select" in buf.lower()):
                time.sleep(0.5)
                os.write(fd, (idx + "\n").encode())
                sent = True
try:
    os.kill(pid, signal.SIGKILL)
except ProcessLookupError:
    pass
os.system("pkill -9 -x memtest_vulkan 2>/dev/null")
