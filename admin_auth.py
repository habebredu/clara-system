from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os

security = HTTPBasic()


def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password

    if username != "admin" or password != "admin":
        raise HTTPException(status_code=401, detail="Unauthorized")

    return username
