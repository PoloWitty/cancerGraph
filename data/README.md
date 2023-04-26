# download graphData
<!-- use `aliyunpan` to download the graphData

1. download `aliyunpan`
```bash
wget https://github.com/tickstep/aliyunpan/releases/download/v0.2.7/aliyunpan-v0.2.7-linux-amd64.zip
unzip aliyunpan-v0.2.7-linux-amd64.zip
cd aliyunpan-v0.2.7-linux-amd64
```

2. use `aliyunpan` to download the graphData
```bash
./aliyunpan download 

``` -->
## use gdown to download data
1. install `gdown` use pip
```bash
pip install gdown

# to upgrade
pip install --upgrade gdown
```
2. download graphData
```bash
gdown --folder https://drive.google.com/drive/folders/1NPG-61qI8IoUQAqdHGAUJUrdpvM1IR30?usp=sharing
```

## More: how to use google drive to upload data

1. download gdrive3 (finish these [steps](https://github.com/glotlabs/gdrive/blob/main/docs/create_google_api_credentials.md) befrore doing this)

note that gdrive2 will report an error

```bash
wget https://github.com/glotlabs/gdrive/releases/download/3.9.0/gdrive_linux-x64.tar.gz
tar -xzvf gdrive_linux-x64.tar.gz
```
2. add google account to gdrive

```bash
./gdrive account add
```