from PIL import Image


def text_preprocess_train_function(sample, num_out_images=None, image_size=None):
    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample["text_input"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["text_output"]},
            ],
        }
    ]

    return conversations


def text_preprocess_test_function(sample):
    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample["text_input"]},
            ],
        },
    ]

    return conversations


def single_input_image_preprocess_function(sample, num_out_images=None, image_size=None):
    # Load images
    image = Image.open(sample["image_input"]).convert("RGB") 

    if image_size is not None:
        # Format conversations
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "resized_height": image_size, "resized_width":image_size},
                    {"type": "text", "text": sample["text_input"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["text_output"]},
                    ],
            }
        ]
    else:
        # Format conversations
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["text_input"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["text_output"]},
                    ],
            }
        ] 

    return conversations


def single_input_image_multi_output_image_preprocess_function(sample, num_out_images=-1, image_size=None):
    # Load images
    image = Image.open(sample["image_input"]).convert("RGB")     
    image_outputs = [Image.open(image_output).convert("RGB") for image_output in sample["image_outputs"]]

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": image_output}
                for image_output in image_outputs
            ] + [{"type": "text", "text": sample["text_output"]}],
        }
    ]

    return conversations


def single_input_image_test_preprocess_function(sample):
    # Load images
    image = Image.open(sample["image_input"]).convert("RGB") 

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
    ]

    return conversations


task_preprocess_functions = {
    "sft_image": single_input_image_preprocess_function,
    "sft_text": text_preprocess_train_function,
}


task_test_preprocess_functions = {
    "sft_image": single_input_image_test_preprocess_function,
    "sft_text": text_preprocess_test_function,
}


ACTION_MAP = {
    "LEFT":  (0, -1),
    "DOWN":  (1,  0),
    "RIGHT": (0,  1),
    "UP":    (-1, 0),
    "L":  (0, -1),
    "D":  (1,  0),
    "R": (0,  1),
    "U":  (-1, 0)
}

def parse_action_sequence(path_str):

    path_str = path_str.upper()
    action_sequence = [char for char in list(path_str) if char in ['U', 'D', 'R', 'L']]

    return action_sequence


def simulate_vsp(map_desc, path_str):
    """
    Simulate the action sequence on the provided map.
    """
    action_sequence = parse_action_sequence(path_str)

    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c)
                break
        if start is not None:
            break

    if start is None:
        raise ValueError("The map description does not contain a start position (cell value 1).")
    
    current_position = start
    for action in action_sequence:
        if action not in ACTION_MAP:
            return {
                "success": False,
                "status": "Invalid action",
                "invalid": True,
            }
            raise ValueError(f"Invalid action: {action}. Valid actions are {list(ACTION_MAP.keys())}.")
        
        dr, dc = ACTION_MAP[action]
        new_r = current_position[0] + dr
        new_c = current_position[1] + dc

        # Check boundaries; ignore the move if it goes out-of-bounds.
        if new_r < 0 or new_r >= len(map_desc) or new_c < 0 or new_c >= len(map_desc[0]):
            continue
        
        current_position = (new_r, new_c)
        
        # If the new cell is a hole (-1), immediately return failure.
        if map_desc[new_r][new_c] == -1:
            return {
                "success": False,
                "status": "Fell in hole",
                "invalid": False,
            }

    # Check if the final position is the goal (2).
    final_r, final_c = current_position
    if map_desc[final_r][final_c] == 2:
        return {
            "success": True,
            "status": "Reached goal",
            "invalid": False,
        }
    else:
        return {
            "success": False,
            "status": "Did not reach goal",
            "invalid": False,
        }