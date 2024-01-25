FROM nvidia/cuda:11.0-base

# prerequisites
RUN DEBIAN_FRONTEND="noninteractive" apt-get update -yqq && apt-get upgrade -yqq
RUN DEBIAN_FRONTEND="noninteractive" apt-get install --fix-missing -yqq git cmake gcc-8 g++-8 g++ libtbb-dev libasio-dev libconfig++ libboost-filesystem-dev autoconf automake libtool curl make unzip libboost-all-dev clang llvm sudo nvidia-cuda-dev 
WORKDIR /

# clone the repositories
ARG repo_username
ARG repo_password

RUN git clone --recursive -b v3.12.0 https://github.com/protocolbuffers/protobuf.git
RUN git clone --recursive -b master https://gitlab.mpi-sws.org/cld/ml/clockwork.git
RUN git clone --depth=1 --recursive --single-branch --branch clockwork-v0.6 https://gitlab.mpi-sws.org/cld/ml/tvm.git
RUN git clone --depth=1 --recursive --single-branch --branch master https://gitlab.mpi-sws.org/cld/ml/clockwork-results.git


ARG ncpus

# make protobuf
WORKDIR /protobuf
RUN ./autogen.sh 
RUN ./configure 
RUN make -j ${ncpus} 
RUN make install
RUN ldconfig

# make tvm
WORKDIR /tvm
RUN cmake .
RUN make -j ${ncpus}
RUN make install


ENV TVM_HOME=/tvm
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TVM_HOME/build
ENV DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$TVM_HOME/build

# make clockwork
WORKDIR /clockwork
#RUN git checkout tags/osdi_2020_submit
RUN mkdir build
WORKDIR /clockwork/build
RUN cmake ..
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
RUN make -j ${ncpus}

# set environment variables
ENV CUDA_CACHE_DISABLE=1
ENV CLOCKWORK_MODEL_DIR=/clockwork-modelzoo-volta
ENV AZURE_TRACE_DIR=/azure-functions

# Get the extra models and trace
### SLOW
#RUN git clone --depth=1 --recursive --single-branch --branch master https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta.git
### FASTER
#RUN apt-get install -yqq wget
#WORKDIR /clockwork-modelzoo-volta
#RUN wget -c https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta/-/archive/master/clockwork-modelzoo-volta-master.tar.gz -O - |tar -zx

### CHEATING FOR NOW
#ADD clockwork-modelzoo-volta/ /clockwork-modelzoo-volta
#ADD azure-functions/ /azure-functions

#RUN git clone --depth=1 --recursive --single-branch --branch master https://${repo_username}:${repo_password}@gitlab.mpi-sws.org/cld-private/datasets/azure-functions.git
#RUN git clone --depth=1 --recursive --single-branch --branch master https://${repo_username}:${repo_password}@gitlab.mpi-sws.org/cld/trace-datasets/azure-functions.git new-azure-functions

# Add ssh user
RUN apt-get install -yqq openssh-server
RUN useradd -s /bin/bash -m clockwork -u 1337
RUN chown -R clockwork.clockwork /clockwork /tvm /protobuf

# Add useful tools to make the environment friendlier
RUN apt-get install -yqq vim nano sudo

# keeps the docker running
#CMD tail -f /dev/null
RUN mkdir -p /run/sshd
RUN echo "PermitUserEnvironment yes" >> /etc/sshd_config
CMD /usr/sbin/sshd -D

