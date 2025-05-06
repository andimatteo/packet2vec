# test_generate_pcap.py
from scapy.all import Ether, IP, TCP, UDP, ICMP, Raw, Dot1Q, wrpcap
import random
import time
# Costruiamo qualche pacchetto di esempio
pkts = []

# 1) ICMP Echo Request
pkts.append(
    Ether() /
    IP(src="10.0.0.1", dst="10.0.0.2") /
    ICMP(type="echo-request") /
    b"hello"
)

# 2) TCP SYN
pkts.append(
    Ether() /
    IP(src="10.0.0.3", dst="10.0.0.4") /
    TCP(sport=12345, dport=80, flags="S") /
    b"GET / HTTP/1.1\r\n\r\n"
)
pkts.append(
    Ether() /
    IP(src="10.0.0.3", dst="10.0.0.4") /
    TCP(sport=12345, dport=80, flags="S") /
    b"GET / HTTP/1.1\r\n\r\n"
)
pkts.append(
    Ether() /
    IP(src="10.0.0.3", dst="10.0.0.4") /
    TCP(sport=12345, dport=80, flags="S") /
    b"GET / HTTP/1.1\r\n\r\n"
)

# 3) UDP packet
pkts.append(
    Ether() /
    IP(src="10.0.0.5", dst="10.0.0.6") /
    UDP(sport=54321, dport=53) /
    b"\x12\x34\x56\x78"
)
# 1) Pacchetto HTTP SYN + GET su VLAN 100 verso porta 80
pkts.append(
    Ether() /
    Dot1Q(vlan=100) /
    IP(src="10.155.15.4", dst="10.0.0.4", ttl=128, id=random.randint(1, 65535)) /
    TCP(sport=4242, dport=80, flags="S", seq=random.randint(0, 2**32-1)) /
    Raw(load=b"GET /index.html HTTP/1.1\r\nHost: example.com\r\nUser-Agent: ScapyBot/1.0\r\n\r\n")
)

# 2) Pacchetto UDP con payload casuale lungo 32 byte verso porta 53 (DNS)
payload = bytes(random.getrandbits(8) for _ in range(32))
pkts.append(
    Ether() /
    IP(src="10.155.15.4", dst="10.1.0.4", ttl=64, tos=0x10) /
    UDP(sport=5353, dport=53) /
    Raw(load=payload)
)

# 3) Pacchetto ICMP echo request con timestamp come payload
timestamp = str(time.time()).encode()
pkts.append(
    Ether() /
    IP(src="10.155.15.4", dst="10.1.0.4", id=0xBEEF, flags="DF") /
    ICMP(type="echo-request", id=0x42, seq=1) /
    Raw(load=timestamp)
)

pkts.append(
    Ether() /
    IP(src="10.155.15.4", dst="10.1.0.4") /
    TCP(sport=4242, dport=80, flags="S") /
    b"GET / HTTP/1.1\r\n\r\n"
)
# Salviamo su file test.pcap
output = "data/test/UDPFlood.pcap"
wrpcap(output, pkts)
print(f"Saved {len(pkts)} packets to {output}")

