FROM rocker/r-ver:4.5.2
# irace depends on fs, which needs the system libuv headers when built from CRAN.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libuv1-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN Rscript --vanilla -e "install.packages('irace', repos='https://cloud.r-project.org'); library(irace); cat('Installed irace', as.character(packageVersion('irace')), '\n')"
ENTRYPOINT ["Rscript", "--vanilla"]
