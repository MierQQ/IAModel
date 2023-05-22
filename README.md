# IAModel

##  Description

It's emotional intelligent assistant model

## How to run

The first, you need to unpack models files in /backend/app in the folder

now you can run docker compose

```
docker compose up -d
```

## Configure

### Nginx config

located in /frontend/webgl.conf

### Frontend config

located in /frontend/webgl/web-config.json

its json containing url to backend api, if you want to host frontend remotely, make sure to change api in file

structure: 

```json
{
    "backend" : string # api to backend
}
```



