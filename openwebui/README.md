# Open WebUI và Pipelines Setup Guide

Hướng dẫn thiết lập và tùy chỉnh Open WebUI kết hợp với Pipelines để tạo ra một hệ thống AI chatbot linh hoạt và có thể mở rộng.

## Khởi động nhanh

### Bước 1: Cài đặt Open WebUI qua Conda
```bash
# Tạo và kích hoạt môi trường conda
conda create -n open-webui python=3.11
conda activate open-webui

# Cài đặt Open WebUI
pip install open-webui
```

### Bước 2: Khởi động OpenWebui
```bash
# Chạy Open WebUI
open-webui serve --port 7000

# Truy cập http://localhost:7000
# Kết nối Pipelines: Admin Panel > Settings > Connections
# API URL: http://localhost:9099, API Key: 0p3n-w3bu!
```


### Bước 3: Chạy Pipelines qua Docker

```bash
cd pipelines
docker stop pipelines
docker rm pipelines
docker run -d -p 9099:9099 --add-host=host.docker.internal:host-gateway -v "${PWD}/pipelines:/app/pipelines" --name pipelines --restart always ghcr.io/open-webui/pipelines:main
```



## Hướng dẫn cho nhà phát triển

### Cấu trúc Pipeline có sẵn
```
openwebui/
└── pipelines/
    ├── main.py              # Server chính
    ├── start.sh             # Script khởi động
    ├── pipelines/           # Thư mục chứa custom pipelines
    ├── examples/            # Ví dụ pipelines
    ├── blueprints/          # Templates
    └── requirements.txt     # Dependencies
```

### Tạo Pipeline tùy chỉnh
```bash
# Tạo pipeline mới trong thư mục pipelines/
nano pipelines/my_custom_pipeline.py
```

### Template Pipeline cơ bản
```python
# pipelines/my_custom_pipeline.py
"""
title: My Custom Pipeline
author: Your Name
date: 2024-01-01
version: 1.0
license: MIT
description: Pipeline tùy chỉnh của tôi
requirements: requests, pandas
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        # Cấu hình pipeline
        custom_param: str = "default_value"
    
    def __init__(self):
        self.name = "My Custom Pipeline"
        self.description = "Mô tả pipeline của bạn"
        self.valves = self.Valves()
    
    async def on_startup(self):
        print(f"🚀 {self.name} started")
    
    async def on_shutdown(self):
        print(f"⏹️ {self.name} stopped")
    
    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        # Logic xử lý của bạn ở đây
        return f"Processed: {user_message}"
```

#### Xem danh sách pipelines
```bash
curl http://localhost:9099/pipelines
```

#### Reload pipelines sau khi chỉnh sửa
```bash
curl -X POST http://localhost:9099/pipelines/reload
```

#### Kiểm tra valves của pipeline
```bash
curl http://localhost:9099/{pipeline_id}/valves
```

#### Kiểm tra logs
```bash
# Xem logs pipelines server
tail -f logs/pipelines.log

# Hoặc chạy với debug mode
GLOBAL_LOG_LEVEL=DEBUG python main.py
```

#### Port conflicts
- Open WebUI: `3000`
- Pipelines: `9099`
- Đảm bảo các port này không bị sử dụng




