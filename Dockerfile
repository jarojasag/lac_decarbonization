FROM julia:1.8.5-bullseye
WORKDIR /sisepuede

# COPY SISEPUEDE COMPONENTS OVER
COPY ./docs ./docs
COPY ./python ./python
COPY ./julia ./julia
COPY ./ref ./ref
COPY ./sisepuede.config ./sisepuede.config

# UPDATE AND GET KEY TOOLS
RUN apt-get update \
    && apt-get install -y curl git g++ gcc build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# INSTALL PYTHON
# POTENTIAL SOLUTION TO ENVIRONMENT ISSUES: https://pythonspeed.com/articles/activate-conda-dockerfile/
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda install python=3.11 pip

# COPY OVER REQUIREMENTS AND INSTALL 

# INSTALL NEMOMOD AND INSTANTIATE JULIA COMPONENTS
#RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate();'

# TEMPORARY CMD
CMD ["bash"]
