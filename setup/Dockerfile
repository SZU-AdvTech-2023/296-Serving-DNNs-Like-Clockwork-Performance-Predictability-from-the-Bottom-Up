FROM registry:5000/clockwork:artifact

# Allow the same ssh access within the container as is on this node
# Note: requires docker to have the whole home directory as its build context
ADD setup/ssh/authorized_keys /home/clockwork/.ssh/authorized_keys
ADD setup/ssh/environment /home/clockwork/.ssh/environment
RUN service ssh start\
    && /usr/bin/ssh-keygen -A \
    && chmod 600 /home/clockwork/.ssh/* \
    && chown 1337 /home/clockwork/.ssh -R

