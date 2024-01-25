FROM clockworkosdi2020ae/clockwork:latest

# Allow the same ssh access within the container as is on this node
# Note: requires docker to have the whole home directory as its build context
ADD .ssh/authorized_keys /home/clockwork/.ssh/authorized_keys
ADD clockwork-results/setup/environment /home/clockwork/.ssh/environment
RUN service ssh start
RUN /usr/bin/ssh-keygen -A
RUN chmod 600 /home/clockwork/.ssh/*
RUN chown 1337 /home/clockwork/.ssh -R

