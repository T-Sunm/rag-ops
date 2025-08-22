# Open WebUI và Pipelines Setup Guide

Hướng dẫn thiết lập và tùy chỉnh Open WebUI kết hợp với Pipelines để tạo ra một hệ thống AI chatbot linh hoạt và có thể mở rộng.

## Mục tiêu

1. **Setup Environment**: Tạo môi trường conda và cài đặt Open WebUI
2. **Customize Pipelines**: Tùy chỉnh pipelines có sẵn theo nhu cầu
3. **Run System**: Chạy hệ thống hoàn chỉnh với các tính năng tùy chỉnh

## 1. Setup Environment

### Tạo Conda Environment
```bash
# Tạo environment mới với Python 3.11
conda create --name openwebui-env python=3.11
conda activate openwebui-env
```

### Cài đặt Open WebUI
```bash
# Cài đặt Open WebUI
pip install open-webui

# Kiểm tra cài đặt
open-webui --help
```

## 2. Customize Pipelines

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

### Tùy chỉnh Dependencies
Nếu bạn muốn thay đổi cách cài đặt dependencies, chỉnh sửa file `start.sh`:

```bash
# Mở file start.sh để chỉnh sửa
nano start.sh

# Tìm dòng 100 và comment nếu không muốn auto-install:
# pip install $requirements
```

### Tạo Custom Pipeline
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

## 3. Run System

### Cài đặt Dependencies cho Pipelines
```bash
# Từ thư mục openwebui/pipelines
cd openwebui/pipelines
pip install -r requirements.txt
```

### Chạy Pipelines Server
```bash
# Sử dụng script start.sh
./start.sh

# Hoặc chạy trực tiếp
python main.py --port 9099
```

### Chạy Open WebUI
```bash
# Terminal mới, active environment
conda activate openwebui-env

# Chạy Open WebUI
open-webui serve --port 3000
```

### Kết nối Open WebUI với Pipelines
1. Mở trình duyệt: `http://localhost:3000`
2. Vào **Settings** > **Connections** > **OpenAI API**
3. Cấu hình:
   - **API Base URL**: `http://localhost:9099/v1`
   - **API Key**: `0p3n-w3bu!` (hoặc key từ config.py)

## Optional: Sử dụng Poetry thay vì Pip

### Cài đặt Poetry
```bash
# Cài đặt Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Hoặc qua pip
pip install poetry
```

### Sử dụng Poetry với Pipelines
```bash
# Trong thư mục pipelines
poetry init
poetry install

# Chạy với Poetry
poetry run python main.py --port 9099
```

### Tùy chỉnh start.sh cho Poetry
Nếu muốn sử dụng Poetry thay vì pip, chỉnh sửa file `start.sh`:

```bash
# Mở file start.sh
nano start.sh

# Tìm và comment dòng 100:
# pip install $requirements

# Thêm dòng mới (tuỳ chọn):
# poetry install
```

## Quản lý Pipelines

### Xem danh sách pipelines
```bash
curl http://localhost:9099/pipelines
```

### Reload pipelines sau khi chỉnh sửa
```bash
curl -X POST http://localhost:9099/pipelines/reload
```

### Kiểm tra valves của pipeline
```bash
curl http://localhost:9099/{pipeline_id}/valves
```

## Debugging và Troubleshooting

### Kiểm tra logs
```bash
# Xem logs pipelines server
tail -f logs/pipelines.log

# Hoặc chạy với debug mode
GLOBAL_LOG_LEVEL=DEBUG python main.py
```

### Port conflicts
- Open WebUI: `3000`
- Pipelines: `9099`
- Đảm bảo các port này không bị sử dụng

### Environment issues
```bash
# Kiểm tra conda environment
conda env list

# Kiểm tra packages đã cài
pip list | grep -E "(openwebui|fastapi|uvicorn)"
```


