#!/usr/bin/env bash
# Container start command for the verl-defaults check on a Runpod pod built from
# lmsysorg/sglang:v0.5.23, which has no SSH server. Passed base64-encoded through
# --docker-args so that no quoting reaches Runpod:
#   --docker-args "bash -c 'echo <base64 of this file> | base64 -d | bash'"
# Installs sshd with the account's injected key (PUBLIC_KEY), starts the model
# download in the background, then keeps the container alive.
mkdir -p /root/.ssh /run/sshd /models
chmod 700 /root/.ssh
echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
nohup python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B', local_dir='/models/Qwen3.5-9B')" \
  > /root/model-download.log 2>&1 &
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync > /root/apt.log 2>&1
ssh-keygen -A
/usr/sbin/sshd && echo "sshd up" >> /root/apt.log
sleep infinity
