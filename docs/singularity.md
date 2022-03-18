## Singulatiry (tested on Ubuntu 20.04)

### Install Go

```bash
# Ensure repositories are up-to-date
sudo apt-get update
# Install debian packages for dependencies
sudo apt-get install -y \
    build-essential \
    libseccomp-dev \
    pkg-config \
    squashfs-tools \
    cryptsetup
```

```bash
export VERSION=1.17.3 OS=linux ARCH=amd64  # change this as you need

wget -O /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz \
  https://dl.google.com/go/go${VERSION}.${OS}-${ARCH}.tar.gz
sudo tar -C /usr/local -xzf /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz

echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```

## Install Singularity

```bash
git clone https://github.com/sylabs/singularity.git
cd singularity
git checkout v3.8.0

./mconfig
make -C builddir
sudo make -C builddir install
```

Reference: [Singularity docs](https://sylabs.io/guides/latest/admin-guide/installation.html#before-you-begin)
