FROM rocker/r-ver:4.5.2
# irace depends on fs, which needs the system libuv headers when built from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libuv1-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
# A dated package snapshot (the one rocker/r-ver:4.5.2 uses), so every build installs the same irace release.
RUN Rscript --vanilla -e "install.packages('irace', repos='https://p3m.dev/cran/__linux__/noble/2026-03-10'); library(irace); cat('Installed irace', as.character(packageVersion('irace')), '\n')"
ENTRYPOINT ["Rscript", "--vanilla"]
