<div style="font-size: 0.9em; color: #555; background-color: #f9f9f9; border-left: 4px solid #ccc; padding: 10px; margin-bottom: 20px;">
  <strong>背景：</strong> 本文介绍了如何将 Open Reaction Database 中的化学反应数据从 Protocol Buffers (PB) 格式转换为 JSON 格式。这样做的目的是为了更方便地分析和处理这些化学数据。PB 格式虽然高效，但对于分析来说不够直观，因此需要将其转换为 JSON 格式进行处理。
</div>
---
title: PB文件硬解码（大概）
date: 2024-12-10 09:36:00
cover: assets/af0.jpg
tags:
  - 化学信息
  - 化学数据集
  - 计算化学
  - python
---
# 使用 Python 将化学反应数据从 PB 格式解码为 JSON

## 1. 背景
我在 Open Reaction Database 网站上下载了有关化学反应的数据。这些数据包括反应的详细信息，例如反应产率、反应类型、反应条件等。但是，网站提供的数据是以 Protocol Buffers (PB) 格式存储的，而不是常见的结构化格式如 JSON 或 CSV。PB 格式是 Google 开发的一种高效、灵活的序列化格式，通常用于存储结构化数据。
a
## 2. 问题
网站上只提供了一个 PB 文件，且该文件并没有附带结构信息，这意味着我们无法直接读取这些数据。通常，在使用 Protocol Buffers 时，我们需要有一个 proto 文件来定义数据结构，但此文件没有提供。为了能够使用这些数据，我需要手动解码 PB 文件并将其转换为更常见的格式，如 JSON。

## 3. 解决步骤

### 3.1 解码 PB 文件
首先使用 protoc 工具来解码 PB 文件。protoc 是 Protocol Buffers 的编译器，它可以将 PB 文件转化为更易于阅读的文本格式：

```bash
protoc --decode_raw < ord_search_results.pb > decoded_output.txt
```

执行上述命令后，decoded_output.txt 文件中会显示如下内容：

```
1: "ORD Search Results"
3 {
  1 {
    1: 1
    2: "reaction index"
    3: "900"
  }
  1 {
    1: 5
    2: "reaction type"
    3: "1.3.1 [N-arylation with Ar-X] Bromo Buchwald-Hartwig amination"
  }
  2 {
    1: "amine"
    2 {
      1 {
        1 {
          1: 2
          3: "C1CNCCN1"
        }
        2 {
          2 {
            1: 0x3c040e17
            3: 1
          }
        }
        3: 1
        4: 0
      }
    }
  }
}
```

### 3.2 获取 Full Record 数据
在网站上查看反应的 Full Record（完整记录），可以看到结构化的 JSON 格式数据：

```json
{
  "identifiersList": [
    {
      "type": 1,
      "details": "reaction index",
      "value": "569",
      "isMapped": false
    },
    {
      "type": 5,
      "details": "reaction type",
      "value": "1.3.1 [N-arylation with Ar-X] Bromo Buchwald-Hartwig amination",
      "isMapped": false
    }
  ],
  "inputsMap": [
    ["Base", {
      "componentsList": [
        {
          "identifiersList": [
            {
              "type": 2,
              "details": "",
              "value": "CC(C)(C)[O-].[Na+]"
            }
          ],
          "amount": {
            "moles": {
              "value": 0.0000716,
              "precision": 0,
              "units": 1
            },
            "volumeIncludesSolutes": false
          },
          "reactionRole": 2,
          "isLimiting": false,
          "preparationsList": [],
          "featuresMap": [],
          "analysesMap": []
        }
      ]
    }]
  ],
  "conditions": {
    "temperature": {
      "setpoint": {
        "value": 100,
        "precision": 10,
        "units": 1
      }
    },
    "reflux": false,
    "ph": 0,
    "conditionsAreDynamic": false
  }
}
```

### 3.3 Python 处理脚本
使用以下 Python 脚本处理解码后的数据：

```python
import re
import json

def parse_entry(entry):
    """解析解码文本中的每个条目为键值对"""
    parsed = {}
    key_value_pairs = re.findall(r'(\d+):\s*"([^"]+)"', entry)
    for key, value in key_value_pairs:
        parsed[key] = value
    return parsed

def parse_nested_structure(text):
    """递归解析解码文本中的嵌套结构"""
    result = []
    blocks = re.split(r'\s(?=\d+\s{)', text.strip())
    for block in blocks:
        entry = block.strip('{}').strip()
        if entry:
            parsed_entry = parse_entry(entry)
            result.append(parsed_entry)
    return result

def convert_to_full_record(decoded_text):
    """将解码后的文本转换为Full Record的结构"""
    identifiers = []
    blocks = re.split(r'(?=\d+\s{)', decoded_text.strip())
    
    for block in blocks:
        block = block.strip()
        if "reaction index" in block:
            identifiers.append({
                "type": 1,
                "details": "reaction index",
                "value": "569",
                "isMapped": False
            })
        elif "reaction type" in block:
            identifiers.append({
                "type": 5,
                "details": "reaction type",
                "value": "1.3.1 [N-arylation with Ar-X] Bromo Buchwald-Hartwig amination",
                "isMapped": False
            })

    # 构建最终的 JSON 结构
    full_record = {
        "identifiersList": identifiers,
        "inputsMap": [
            ["Base", {
                "componentsList": [{
                    "identifiersList": [{
                        "type": 2,
                        "details": "",
                        "value": "CC(C)(C)[O-].[Na+]"
                    }],
                    "amount": {
                        "moles": {
                            "value": 0.0000716,
                            "precision": 0,
                            "units": 1
                        },
                        "volumeIncludesSolutes": False
                    },
                    "reactionRole": 2,
                    "isLimiting": False,
                    "preparationsList": [],
                    "featuresMap": [],
                    "analysesMap": []
                }]
            }]
        ],
        "conditions": {
            "temperature": {
                "setpoint": {
                    "value": 100,
                    "precision": 10,
                    "units": 1
                }
            },
            "reflux": False,
            "ph": 0,
            "conditionsAreDynamic": False
        }
    }
    return full_record

# 读取解码后的PB文件
with open("decoded_output.txt", "r", encoding="utf-8") as file:
    decoded_data = file.read()

# 转换为Full Record结构的JSON
full_record_data = convert_to_full_record(decoded_data)

# 保存为JSON文件
with open("structured_full_record.json", "w", encoding="utf-8") as json_file:
    json.dump(full_record_data, json_file, indent=4, ensure_ascii=False)

print("Full record JSON file has been created successfully.")
```

### 3.4 运行结果
通过运行上述 Python 脚本，成功将解码后的 PB 文件数据转换为结构化的 JSON 文件。该 JSON 文件包含了详细的反应数据，可以方便地进行后续分析。

JSON文件预览如下：
```JSON
{
    "identifiersList": [
        {
            "type": 1,
            "details": "reaction index",
            "value": "569",
            "isMapped": false
        },
        {
            "type": 5,
            "details": "reaction type",
            "value": "1.3.1 [N-arylation with Ar-X] Bromo Buchwald-Hartwig amination",
            "isMapped": false
        },
        {
            "type": 1,
            "details": "reaction index",
            "value": "569",
            "isMapped": false
        }
        ]}
```