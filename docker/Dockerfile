# Dockerfile for COSItools
# 
# Build the docker with:
# docker build -t cosi-main - < Dockerfile

FROM ubuntu:22.04

MAINTAINER Andreas Zoglauer <zoglauer@berkeley.edu>

# Install all the COSItools prerequisites
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq gosu vim nano less gzip git git-lfs gawk dpkg-dev make g++ gcc gfortran gdb valgrind binutils libx11-dev libxpm-dev libxft-dev libxext-dev libssl-dev libpcre3-dev libglu1-mesa-dev libglew-dev libftgl-dev libmysqlclient-dev libfftw3-dev libgraphviz-dev libavahi-compat-libdnssd-dev libldap2-dev python3 python3-pip python3-dev python3-tk python3-venv libxml2-dev libkrb5-dev libgsl-dev cmake libxmu-dev curl doxygen libblas-dev liblapack-dev expect dos2unix libncurses5-dev libboost-all-dev libcfitsio-dev mlocate libxerces-c-dev libhealpix-cxx-dev libbz2-dev wget

# Add COSI user
RUN groupadd -g 1111 cosi && useradd -u 1111 -g 1111 -ms /bin/bash cosi
#RUN ln -s /usr/bin/python3 /usr/bin/python

# Switch to user cosi
USER cosi

# Setup 
#RUN cd /home/cosi && /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/cositools/cosi-setup/main/setup.sh)"
#RUN echo . /home/cosi/COSItools/source.sh >> /home/cosi/.bashrc
#switch megalib branch
# === MEGAlib ===

# Needs a lot of memory. Check your docker preferences if it fails

RUN cd /home/cosi/ && git clone https://github.com/zoglauer/megalib && cd megalib && git checkout develop-cosi

#COPY build-root.sh $MEGALIBDIR/config/build-root.sh

RUN cd /home/cosi/megalib && \
     bash config/configure_packages && \
    bash setup.sh --branch=develop-cosi && \
    echo . /home/cosi/megalib/bin/source-megalib.sh >> $HOME/.bashrc

#ENV PATH="/root/.local/bin:${PATH}"

RUN cd /home/cosi && python3 -m venv bct

RUN cd /home/cosi \
      && git clone https://gitlab.com/burstcube/bc-tools.git \
      && . bct/bin/activate  \
      && cd bc-tools && git checkout develop && pip install . && cd ..\
      && wget https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/gbm_data_tools/gbm_data_tools-1.1.1.tar.gz  \
      && python3 -m pip install  basemap ipykernel "matplotlib>=3.7.1" \
      && pip3 install ./gbm_data_tools-1.1.1.tar.gz \
      && python -m ipykernel install --user --name=bct \
      && sed -i 's/trapz/trapezoid/g' bct/lib/python3.10/site-packages/gdt/core/data_primitives.py 


RUN cd /home/cosi && python3 -m venv deeplearning

RUN cd /home/cosi \
    . deeplearning/bin/activate  \
    && python3 -m pip install tensorflow[and-cuda]==2.19.0 keras matplotlib scipy plotly mhealpy pandas astropy scikit-learn seaborn ipykernel  \
    && python3 -m ipykernel install --user --name=deeplearning 

RUN cd /home/cosi && python3 -m venv cosipy

RUN cd /home/cosi \
    . cosipy/bin/activate  \
    && python3 -m pip install --use-pep517 cosipy \
    && python3 -m pip install ipykernel \
    && python3 -m ipykernel install --user --name=cosipy


RUN pip3 install --user pytest-cov

RUN pip3 install --user sphinx sphinx_rtd_theme

RUN pip3 install --user mhealpy

#To build tutorials
RUN pip3 install --user nbsphinx jupyter_client ipykernel
#ADD --chown=user:user https://github.com/jgm/pandoc/releases/download/2.9.2.1/pandoc-2.9.2.1-linux-amd64.tar.gz $HOME
RUN cd $HOME && wget https://github.com/jgm/pandoc/releases/download/2.9.2.1/pandoc-2.9.2.1-linux-amd64.tar.gz
RUN tar xvzf $HOME/pandoc-2.9.2.1-linux-amd64.tar.gz --strip-components 1 -C $HOME/.local

# For full instrument response
RUN pip3 install --user h5py

#RUN cd /home/cosi && git clone https://github.com/cositools/grb-simulations.git


# Switch back to ROOT
USER root

# For full instrument response
RUN pip3 install notebook pandas

# Create entry-point script - changes the UID of cosi to the local USER and group for full access to the exchange directory on all machines
RUN    cd /usr/local/bin \
    && echo '#!/bin/bash' >> entrypoint.sh \
    && echo 'if [ "${USERID}" != "" ]; then usermod -u ${USERID} cosi; fi' >> entrypoint.sh \
    && echo 'if [ "${GROUPID}" != "" ]; then groupmod -g ${GROUPID} cosi; fi' >> entrypoint.sh \
    && echo 'if [ "${USERID}" != "" ] || [ "${GROUPID}" != "" ]; then chown -R cosi:cosi /home/cosi; fi' >> entrypoint.sh \
    && echo ". /home/cosi/megalib/bin/source-megalib.sh" >> entrypoint.sh \
    && echo 'gosu cosi tail -f /dev/null' >> entrypoint.sh \
    && chmod a+rx /usr/local/bin/entrypoint.sh

# The working directory is the COSItools directory
WORKDIR /home/cosi

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]