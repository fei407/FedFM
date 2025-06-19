# Federated Learning with NVIDIA Jetson and Server Setup

This guide provides comprehensive instructions for flashing Jetson devices, setting up Docker environments, configuring server environments, networking, and running FLWR-based federated learning code.

## 0. Flash Jetson OS Image

**Required Hardware:**

- NVIDIA Jetson Orin Nano Super Developer Kit or Jetson AGX Orin Developer Kit
- microSD card (minimum 64 GB)
- DP‑to‑HDMI adapter
- Host computer: Ubuntu 20.04 / 22.04 / 24.04 or Windows with VMware Workstation Pro

**Steps**

1. On Linux, download and install [NVIDIA SDK Manager](https://developer.nvidia.com/nvidia-sdk-manager).
2. Ensure the VMware USB interface is set to **USB 3.1**.
3. Flash the OS by following the instructions from [Jetson AI LAB](https://www.jetson-ai-lab.com/initial_setup_jon.html).

## 1. Set Up Docker on Jetson

### Install Prerequisites

```bash
sudo apt update
sudo apt install -y python3-pip nvidia-jetpack docker.io
sudo pip3 install jetson-stats
```

### Build the Jetson‑Containers Stack

```bash
mkdir workspace && cd workspace
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
./install.sh
```

### Configure Docker Runtime

Edit `/etc/docker/daemon.json`:

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

### Add Swap Space (Jetson Orin Nano Only)

```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /mnt/16GB.swap
sudo mkswap /mnt/16GB.swap
sudo swapon /mnt/16GB.swap
```

### Add User to Docker Group

```bash
sudo usermod -aG docker $USER
```

### Build or Pull a PyTorch Image

```bash
jetson-containers build pytorch:2.6
# Or pull an existing image
docker pull fw407/fedfm
```

## 2. Set Up the Python Environment on the Server

```bash
git clone https://github.com/fei407/FedFM.git
cd FedFM
conda create -n fedfm python=3.10
conda activate fedfm
pip install -r requirements-server.txt
```

## (Opt). Re-build image for fedfm-dev

re-build from `Dockerfile`

```bash
cd FedFM && vim Dockerfile
```

`Dockerfile` :

```bash
FROM fw407/2.6-r36.4-cu126-22.04

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
```

re-build from `Dockerfile`

```bash
docker build -t fedfm-dev .
```

```bash
docker login
docker tag fedfm-dev fw407/fedfm-dev:1.0
docker push fw407/fedfm-dev:1.0
docker images
```

## 3. Configure Networking (Windows + WSL)

### Remove Existing Port‑Proxy Rules (Admin PowerShell)

```powershell
netsh interface portproxy delete v4tov4 listenport=9091 listenaddress=0.0.0.0
netsh interface portproxy delete v4tov4 listenport=9092 listenaddress=0.0.0.0
netsh interface portproxy delete v4tov4 listenport=9093 listenaddress=0.0.0.0
```

### Add New Port‑Proxy Rules

Replace `172.25.41.77` with the WSL IP address:

```powershell
netsh interface portproxy add v4tov4 listenport=9091 listenaddress=0.0.0.0 connectport=9091 connectaddress=172.25.41.77
netsh interface portproxy add v4tov4 listenport=9092 listenaddress=0.0.0.0 connectport=9092 connectaddress=172.25.41.77
netsh interface portproxy add v4tov4 listenport=9093 listenaddress=0.0.0.0 connectport=9093 connectaddress=172.25.41.77
```

### Open Firewall Rules

```powershell
New-NetFirewallRule -DisplayName "ICMPv4-In" -Name "ICMPv4-In" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow
New-NetFirewallRule -DisplayName "Open 9091" -Direction Inbound -LocalPort 9091 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Open 9092" -Direction Inbound -LocalPort 9092 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Open 9093" -Direction Inbound -LocalPort 9093 -Protocol TCP -Action Allow
```

### Verify Connectivity

```bash
# From Jetson
ping <Windows_IP>
nc -zv <Windows_IP> 9092
```

## 4. Run FLWR Components

### Server

```bash
flower-superlink --insecure
```

Open another terminal:

```bash
flwr run . local-deployment --stream
```

### Client (Jetson Docker)

```bash
docker run -it --rm \
  -v /home/fw407/workspace/FedFM:/app/FedFM \
  -w /app/FedFM \
  fedfm-dev bash
```

Inside the container:

```bash
flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config "partition-id=0 num-partitions=10"
```



## License

This project is licensed under the MIT License. See `LICENSE` for details.
