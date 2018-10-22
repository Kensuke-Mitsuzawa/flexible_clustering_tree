FROM frolvlad/alpine-glibc:alpine-3.6
MAINTAINER kensuke-mi <kensuke.mit@gmail.com>

# Mecab install
ENV MECAB_VERSION 0.996
ENV IPADIC_VERSION 2.7.0-20070801
ENV mecab_url https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7cENtOXlicTFaRUE
ENV ipadic_url https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7MWVlSDBCSXZMTXM
ENV build_deps 'curl git bash file sudo openssh gcc make build-base'
ENV dependencies 'openssl'

ENV PATH=/opt/conda/bin:$PATH \
    LANG=C.UTF-8 \
    MINICONDA=Miniconda3-latest-Linux-x86_64.sh
# apk update
RUN apk update

RUN apk add --update --no-cache ${build_deps} \
  # Install dependencies
  && apk add --update --no-cache ${dependencies} \
  # Install MeCab
  && curl -SL -o mecab-${MECAB_VERSION}.tar.gz ${mecab_url} \
  && tar zxf mecab-${MECAB_VERSION}.tar.gz \
  && cd mecab-${MECAB_VERSION} \
  && ./configure --enable-utf8-only --with-charset=utf8 \
  && make \
  && make install \
  && cd \
  # Install IPA dic
  && curl -SL -o mecab-ipadic-${IPADIC_VERSION}.tar.gz ${ipadic_url} \
  && tar zxf mecab-ipadic-${IPADIC_VERSION}.tar.gz \
  && cd mecab-ipadic-${IPADIC_VERSION} \
  && ./configure --with-charset=utf8 \
  && make \
  && make install \
  && cd \
  # Install Neologd
  && git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
  && mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y \
  # Clean up
  && apk del ${build_deps} \
  && rm -rf \
    mecab-${MECAB_VERSION}* \
    mecab-${IPADIC_VERSION}* \
    mecab-ipadic-neologd

# general
RUN apk --no-cache add vim \
wget \
lsof \
curl \
bash \
swig \
gcc \
build-base \
make \
python-dev \
py-pip \
jpeg-dev \
zlib-dev \
git \
linux-headers
ENV LIBRARY_PATH=/lib:/usr/lib

# Python
RUN apk add --no-cache bash wget && \
    wget -q --no-check-certificate https://repo.continuum.io/miniconda/$MINICONDA && \
    bash /Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    ln -s /opt/conda/bin/* /usr/local/bin/ && \
    rm -rf /root/.[acpw]* /$MINICONDA /opt/conda/pkgs/*

RUN conda config --add channels conda-forge --system
RUN conda install Cython \
line_profiler \
pandas \
scikit-learn \
gensim \
scipy \
numpy \
pillow \
matplotlib \
opencv

RUN pip install more_itertools \
jaconv \
joblib \
sqlitedict \
hdbscan \
nltk \
jinja2 \
pyexcelerate \
JapaneseTokenizer \
neologdn


CMD ["/bin/bash"]