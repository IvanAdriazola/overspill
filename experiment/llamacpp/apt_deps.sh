export DEBIAN_FRONTEND=noninteractive
apt-get install -y libssl-dev pkg-config 2>&1 | tail -3
