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

### 1.1 Install Prerequisites

```bash
sudo apt update
sudo apt install -y python3-pip nvidia-jetpack docker.io
sudo pip3 install jetson-stats
```

### 1.2 Build the Jetson‑Containers Stack

```bash
mkdir workspace && cd workspace
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
./install.sh
```

### 1.3 Configure Docker Runtime

Edit `sudo vim /etc/docker/daemon.json`:

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

```bash
sudo systemctl restart docker
sudo docker info | grep 'Default Runtime'
```


### 1.4 Add Swap Space (Jetson Orin Nano Only)

```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /mnt/16GB.swap
sudo mkswap /mnt/16GB.swap
sudo swapon /mnt/16GB.swap
```

### 1.5 Disabling the Desktop GUI

```bash
$ sudo init 3     # stop the desktop
# log your user back into the console (Ctrl+Alt+F1, F2, ect)
$ sudo init 5     # restart the desktop

$ sudo systemctl set-default multi-user.target     # disable desktop on boot
$ sudo systemctl set-default graphical.target      # enable desktop on boot
```

### 1.6 Add User to Docker Group

```bash
sudo usermod -aG docker $USER
```

Then close/restart your terminal (or logout) and you should be able to run docker commands (like `docker info`) without needing sudo.

### 1.7 Build or Pull a PyTorch Image

```bash
jetson-containers build pytorch:2.6
# Or pull an existing image
docker pull fw407/fedfm-dev:1.0
```

## 2. Set Up Docker on Raspberry Pi

```
sudo apt install -y vim
sudo apt install -y htop
mkdir workspace && cd workspace
```

```
sudo apt update && sudo apt upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker $USER
newgrp docker

docker version
```

## 3. Set Up the Python Environment on the Server

```bash
git clone https://github.com/fei407/FedFM.git
cd FedFM
conda create -n fedfm python=3.10
conda activate fedfm
pip install -r requirements-server.txt
```

## 4. Set Up the Python Environment on the Edge devices

```bash
git clone https://github.com/fei407/FedFM.git
cd FedFM
```

## (Opt). Re-build image for fedfm-dev

re-build from `Dockerfile`

```bash
cd FedFM && vim Dockerfile.jetson
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

## (Opt). Re-build image for fedfm-dev

re-build from `Dockerfile`

```bash
cd FedFM && vim Dockerfile.rpi
```

```bash
docker build -f Dockerfile.rpi -t fedfm-rpi:latest .
docker tag fedfm-rpi:latest fw407/fedfm-rpi:latest
docker push fw407/fedfm-rpi:latest

docker pull fw407/fedfm-rpi
```

## 4. Configure Networking (Windows + WSL)

### 4.1 Remove Existing Port‑Proxy Rules (Admin PowerShell)

```powershell
netsh interface portproxy delete v4tov4 listenport=9091 listenaddress=0.0.0.0
netsh interface portproxy delete v4tov4 listenport=9092 listenaddress=0.0.0.0
netsh interface portproxy delete v4tov4 listenport=9093 listenaddress=0.0.0.0
```

### 4.2 Add New Port‑Proxy Rules

Replace `172.25.41.77` with the WSL IP address:

```powershell
netsh interface portproxy add v4tov4 listenport=9091 listenaddress=0.0.0.0 connectport=9091 connectaddress=172.25.41.77
netsh interface portproxy add v4tov4 listenport=9092 listenaddress=0.0.0.0 connectport=9092 connectaddress=172.25.41.77
netsh interface portproxy add v4tov4 listenport=9093 listenaddress=0.0.0.0 connectport=9093 connectaddress=172.25.41.77
```

### 4.3 Open Firewall Rules

```powershell
New-NetFirewallRule -DisplayName "ICMPv4-In" -Name "ICMPv4-In" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow
New-NetFirewallRule -DisplayName "Open 9091" -Direction Inbound -LocalPort 9091 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Open 9092" -Direction Inbound -LocalPort 9092 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Open 9093" -Direction Inbound -LocalPort 9093 -Protocol TCP -Action Allow
```

### 4.4 Verify Connectivity on Edge devices

```bash
# From Jetson
ping 192.168.0.11
nc -zv 192.168.0.11 9092
```

## 5. Run FLWR Components

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
  -p 9034:9034 \
  fw407/fedfm-dev:1.0 \
  flower-supernode \
    --insecure \
    --superlink 192.168.0.11:9092 \
    --clientappio-api-address 0.0.0.0:9034 \
    --node-config "partition-id=4 num-partitions=10"
```

```
docker run -it --rm \
  -v /home/fw407/workspace/FedFM:/app/FedFM \
  -w /app/FedFM \
  -p 9041:9041 \
  fw407/fedfm-rpi \
  flower-supernode \
    --insecure \
    --superlink 192.168.0.11:9092 \
    --clientappio-api-address 0.0.0.0:9041 \
    --node-config "partition-id=5 num-partitions=10"
```



# FedFM Device IP Mapping Table

| Device No. | Device Name | IP Address   |
| ---------- | ----------- | ------------ |
| 0          | 5CG4034DYL  | 192.168.0.11 |
| 1          | agx-orin    | 192.168.0.21 |
| 2          | orin-nano-1 | 192.168.0.31 |
| 3          | orin-nano-2 | 192.168.0.32 |
| 4          | orin-nano-3 | 192.168.0.33 |
| 5          | orin-nano-4 | 192.168.0.34 |
| 6          | rpi-5-1     | 192.168.0.41 |
| 7          | rpi-5-2     | 192.168.0.42 |
| 8          | rpi-5-3     | 192.168.0.43 |
| 9          | rpi-5-4     | 192.168.0.44 |
| 10         | rpi-5-5     | 192.168.0.45 |



## License

This project is licensed under the MIT License. See `LICENSE` for details.

