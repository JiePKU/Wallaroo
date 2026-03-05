IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
TIME_STEP_TOKEN = "<|time_step|>"

DEFAULT_GENERATE_IMAGE_TOKEN = "<|generate_image_pad|>"
GENERATE_START_TOKEN = "<|generate_start|>"
GENERATE_END_TOKEN = "<|generate_end|>"

PIXEL_START_TOKEN = "<|pixel_start|>"
PIXEL_END_TOKEN = "<|pixel_end|>"


hw_indicator_256_lst = ['<indicator:176>', '<indicator:192>', '<indicator:208>', '<indicator:224>', '<indicator:240>', '<indicator:256>', '<indicator:272>', '<indicator:288>', '<indicator:304>', '<indicator:320>', '<indicator:336>', '<indicator:352>']


hw_indicator_384_lst = ['<indicator:256>', '<indicator:288>', '<indicator:320>', '<indicator:336>', '<indicator:352>', '<indicator:384>', '<indicator:416>', '<indicator:432>', '<indicator:448>', '<indicator:480>', '<indicator:512>']


row_col_indicator_384_lst = ['<indicator:16>', '<indicator:18>', '<indicator:20>', '<indicator:21>', '<indicator:22>', '<indicator:24>', '<indicator:26>', '<indicator:27>', '<indicator:28>', '<indicator:30>', '<indicator:32>']


hw_indicator_512_lst = ['<indicator:352>', '<indicator:640>', '<indicator:384>', '<indicator:672>', '<indicator:704>', '<indicator:416>', '<indicator:608>', '<indicator:576>', '<indicator:448>', '<indicator:480>', '<indicator:544>', '<indicator:512>']

row_col_indicator_512_lst = ['<indicator:32>', '<indicator:34>', '<indicator:36>', '<indicator:38>', '<indicator:40>', '<indicator:42>', '<indicator:44>', '<indicator:22>', '<indicator:24>', '<indicator:26>', '<indicator:28>', '<indicator:30>']


EOL_TOKEN = "<|end_of_line>|"

SYSTEM_MESSAGE = "You are a helpful assistant."

# 定义每个字段的采样概率
field_probabilities = {
                            'internvl2_caption_v0': 0.2,
                            'internvl2_caption_v1_en': 0.2,
                            'internvl2_caption_v1_cn': 0.2,
                            'caption_v3_qwen2vl_keywords_cn': 0.1,
                            'caption_v3_qwen2vl_keywords_en': 0.1,
                            'caption_v3_qwen2vl_subject_cn': 0.1,
                            'caption_v3_qwen2vl_subject_en': 0.1,
                            'caption_v3_qwen2vl_overview_cn': 0.3,
                            'caption_v3_qwen2vl_overview_en': 0.3,
                            'caption_v3_intern2VL_detail_cn': 0.3,
                            'caption_v3_intern2VL_detail_en': 0.3,
                            'content': 0.4
                        }

ASPECT_RATIO_1024 = {
    '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.]
}


ASPECT_RATIO_512 = {
     '0.5': [352, 704], '0.52': [352, 672],
     '0.57': [384, 672], '0.6': [384, 640], '0.68': [416, 608], '0.72': [416, 576],
     '0.78': [448, 576], '0.82': [448, 544], '0.88': [480, 544], '0.94': [480, 512],
     '1.0': [512, 512], '1.13': [544, 480], '1.21': [544, 448],
     '1.29': [576, 448], '1.38': [576, 416], '1.46': [608, 416], '1.67': [640, 384],
     '1.75': [672, 384], '2.0': [704, 352]
     }

SigLip_ASPECT_RATIO_512 = {
    '0.52': [364, 700], '0.54': [364, 672], 
    '0.58': [392, 672], '0.61': [392, 644], '0.68': [420, 616], 
    '0.71': [420, 588], '0.76': [448, 588], '0.84': [448, 532], '0.89': [476, 532], '0.94': [476, 504], 
    '1.0': [504, 504], '1.06': [504, 476], '1.12': [532, 476], '1.19': [532, 448],
    '1.31': [588, 448], '1.4': [588, 420], '1.47': [616, 420], '1.64': [644, 392], 
    '1.71': [672, 392], '1.92': [700, 364]}
     

ASPECT_RATIO_384 = {
    '0.5': [256, 512], '0.56': [288, 512], 
    '0.6': [288, 480], '0.71': [320, 448], '0.74': [320, 432], '0.78': [336, 432], 
    '0.8': [336, 416],  '0.85': [352, 416], '0.92': [352, 384], 
    '1.0': [384, 384], 
    '1.09': [384, 352], '1.18': [416, 352],  '1.24': [416, 336], 
    '1.29': [432, 336], '1.35': [432, 320], '1.4': [448, 320], '1.67': [480, 288], 
    '1.78': [512, 288], '2.0': [512, 256]
}


ASPECT_RATIO_256 = {
     '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
     '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
     '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
     '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
     '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
     '1.75': [336.0, 192.0], '2.0': [352.0, 176.0]
}

chinese_template = ['图中有文字:""','图片上写着""','文字""在图中','文字""在图上','图片上有""的字样','文字""在图上','图片上有""的字样']
english_template = ['The picture contains the text ""','Text "" is shown in the image','The words "" appear on the picture','The pattern imprints ""','The image marks ""','"" is displayed on the picture','The words "" are on the picture']

chinese_generate_template = [
    '生成一幅图像，分辨率为{target_w}x{target_h}，基于文本提示：“{prompt}”',
    '依据文本提示：“{prompt}”，创造出一个{target_w}x{target_h}大小的图像',
    '从提示信息：“{prompt}”，制作一幅尺寸为{target_w}x{target_h}的画面',
    '设计一个图像，使用提示『{prompt}』，目标分辨率为{target_w}x{target_h}',
    '利用描述：“{prompt}”，生成分辨率为{target_w}x{target_h}的图片',
    '产生一个图像，尺寸为{target_w}x{target_h}，以文本：“{prompt}”为基础',
    '根据提示："{prompt}"，构建一个{target_w}x{target_h}像素的视觉图片',
    '创建一幅{target_w}x{target_h}的画，用于表达提示：“{prompt}”',
    '以文本提示：“{prompt}”，制作一个分辨率{target_w}x{target_h}的图片',
    '让我们通过输入：“{prompt}”，设计一幅{target_w}x{target_h}大小的图像',
]

english_generate_template = [
    'Generate an image with resolution {target_w}x{target_h} based on the prompt: "{prompt}"',
    'Create a {target_w}x{target_h} sized image using the following text: "{prompt}"',
    'Produce a graphic measuring {target_w}x{target_h} pixels from the description: "{prompt}"',
    'Design an image using the prompt "{prompt}" with a resolution of {target_w}x{target_h}',
    'Craft a picture of resolution {target_w}x{target_h} using the provided prompt: "{prompt}"',
    'Formulate a visual piece of {target_w}x{target_h} dimensions, inspired by: "{prompt}"',
    'Based on the prompt: "{prompt}", create an image with {target_w}x{target_h} pixels',
    'Construct a {target_w}x{target_h} picture to represent the idea of: "{prompt}"',
    'Utilize the text "{prompt}" to design an image with a resolution of {target_w}x{target_h}',
    'Let\'s create a {target_w}x{target_h} image using the following input: "{prompt}"',
]

t2i_chinese_generate_template = [
    '生成一幅图像，基于文本提示：“{prompt}”',
    '依据文本提示：“{prompt}”，创造出一个图像',
    '从提示信息：“{prompt}”制作一幅画面',
    '设计一个图像，使用提示『{prompt}』',
    '利用描述：“{prompt}”生成图片',
    '产生一个图像，以文本：“{prompt}”为基础',
    '根据提示："{prompt}"，构建一个视觉图片',
    '创建一幅画，用于表达提示：“{prompt}”',
    '以文本提示：“{prompt}”制作一个图片',
    '让我们通过输入：“{prompt}”设计一幅图像',
]

t2i_english_generate_template = [
    'Generate an image based on the prompt: "{prompt}"',
    'Create an image using the following text: "{prompt}"',
    'Produce a graphic from the description: "{prompt}"',
    'Design an image using the prompt "{prompt}"',
    'Craft a picture using the provided prompt: "{prompt}"',
    'Based on the prompt: "{prompt}", create an image',
    'Construct a picture to represent the idea of: "{prompt}"',
    'Utilize the text "{prompt}" to design an image',
    'Let\'s create an image using the following input: "{prompt}"',
]

chinese_img2img_template = [
    '根据输入图片，并结合文本提示：“{prompt}”，对图片进行编辑，生成一幅{target_w}x{target_h}的图像',
    '请以给定图片为基础，依据提示：“{prompt}”，创作一个{target_w}x{target_h}的新版图像',
    '在原图的基础上，按照描述：“{prompt}”，生成尺寸为{target_w}x{target_h}的新图',
    '使用输入图片，并参考提示『{prompt}』，对图像进行修改，目标分辨率为{target_w}x{target_h}',
    '结合图片和文本：“{prompt}”，产生一张分辨率为{target_w}x{target_h}的编辑后图片',
    '对给定图片进行调整，依据提示信息：“{prompt}”，输出{target_w}x{target_h}的图像',
    '以输入图片为蓝本，结合提示："{prompt}"，制作一幅{target_w}x{target_h}像素的新画面',
    '基于现有图片和文本：“{prompt}”，创作分辨率为{target_w}x{target_h}的视觉作品',
    '请根据图片内容和提示：“{prompt}”，生成尺寸为{target_w}x{target_h}的图片',
    '让我们以输入图片和文本：“{prompt}”为依据，设计一幅{target_w}x{target_h}的图像',
]

english_img2img_template = [
    'Edit the input image based on the text prompt: "{prompt}", and generate a new image with a resolution of {target_w}x{target_h}.',
    'Using the provided image as a base, create a new {target_w}x{target_h} image according to the prompt: "{prompt}".',
    'Modify the original image following the description: "{prompt}", and produce a new image sized {target_w}x{target_h}.',
    'Take the input image and refer to the prompt: "{prompt}" to edit the image, targeting a resolution of {target_w}x{target_h}.',
    'Combine the image and the text: "{prompt}" to generate an edited picture with a resolution of {target_w}x{target_h}.',
    'Adjust the given image according to the prompt: "{prompt}", and output an image of size {target_w}x{target_h}.',
    'Use the input image as a blueprint, together with the prompt: "{prompt}", to create a new {target_w}x{target_h} visual.',
    'Based on the existing image and the text: "{prompt}", create a visual work with a resolution of {target_w}x{target_h}.',
    'Please generate an image of size {target_w}x{target_h} according to the content of the image and the prompt: "{prompt}".',
    'Let’s design a {target_w}x{target_h} image using the input image and the text prompt: "{prompt}".',
]


english_timestep_template = 'the timestep is {time_step}'
chinese_timestep_template = '时间步为{time_step}'


chinese_response_template = ['这是我为您准备的图像：']
english_response_template = ["Here is the image I've prepared for you:"]

chinese_caption_mmu_template = ['描述一下这幅画面', '请解释这张图片', '你能讲讲这幅图吗？', '这幅图像中有什么内容？', '能详细说明这张照片吗？', '请给出这张图片的描绘', '介绍一下这幅画的内容', '请给我这幅画面的描述', '请详细描述下这张图片']
english_caption_mmu_template = ['Describe this image', 'Please explain the picture', 'Can you talk about the photo?', 'What is in this image?', 'Can you detail the contents of this photo?', 'Provide a depiction of this picture', 'Elaborate on the details of this image', 'Provide a description of this scene']