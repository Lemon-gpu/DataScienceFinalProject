from docx import Document
import os
import base64

def docx_to_markdown(docx_path, md_path, images_dir="images"):
    # 确保图片目录存在
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    doc = Document(docx_path)
    markdown_lines = []
    
    # 计数器，用于给导出的图片文件命名
    image_count = 0

    # 遍历文档中的所有段落
    # 提示：某些元素如图片也可能存在于表格单元格等，这里仅示例处理文档中的段落。
    # 若需处理表格等结构，可扩展逻辑。
    for paragraph in doc.paragraphs:
        md_line_parts = []
        # 遍历该段落的所有run
        for run in paragraph.runs:
            # 获取run底层XML元素，以确认是否包含Drawing(图片)
            inline = run._element
            # w:drawing 元素表示图形/图片
            drawings = inline.xpath('.//w:drawing')
            if drawings:
                # 在drawing元素中寻找图片关系ID
                # 找出blip元素，不使用namespaces参数
                blips = inline.xpath('.//*[local-name()="blip"]')
                for b in blips:
                    rId = b.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                    image_part = doc.part.related_parts[rId]
                    image_data = image_part.blob
                    image_count += 1
                    image_ext = os.path.splitext(image_part.partname)[1]
                    image_filename = f"image_{image_count}{image_ext}"
                    image_path = os.path.join(images_dir, image_filename)

                    with open(image_path, 'wb') as f:
                        f.write(image_data)

                    md_line_parts.append(f"![image]({os.path.join('images', image_filename)})")

            else:
                # 如果不是图片，就直接添加文本
                md_line_parts.append(run.text)
        
        # 合并该段落的markdown文本
        md_line = "".join(md_line_parts).strip()
        # 在markdown中空行一般表示段落分隔，这里简单处理
        if md_line:
            markdown_lines.append(md_line)
        else:
            # 空段落
            markdown_lines.append("")
    
    # 写出最终markdown文件
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(markdown_lines))

# 示例调用
input: str = 'assets/Paper.docx'
output: str = 'assets/Paper.md'
images_dir: str = 'assets/images'
docx_to_markdown(input, output, images_dir)