''' This file creates the dataset used for the whole project.'''
from datasets import load_dataset, load_from_disk, Dataset
import random
from datasets import Dataset
from functools import partial


def extract_first_caption(example):
    """
    Extracts the first caption from the 'txt' field of an example and adds it to the 'caption' field.
    Args:
        example (HuggingFace dataset): A single data example from the dataset.
    Returns:
        dict: Example with the first caption added to the 'caption' field.
    """
    example["caption"] = example["txt"].split("\n")[0]
    return example

def create_matching_pair(example):
     """
    Creates a matching pair (image and its correct caption) with a label of 1.
    Args:
        example (dict): A single data example with an image and a caption.
    Returns:
        dict: A dictionary containing the image, its correct caption, and the label (1).
    """
    return {"image": example["jpg"], "caption": example["caption"], "label": 1}

def create_mismatched_pair(example, index, captions):
     """
    Creates a mismatched pair (image and a random unrelated caption) with a label of 0.
    Args:
        example (HuggingFace dataset): A single data example with an image.
        index (int): The index for selecting a random caption from the captions list.
        captions (list): A list of captions from the dataset.
    Returns:
        dict: A dictionary containing the image, a mismatched caption, and the label (0).
    """
    return {"image": example["jpg"], "caption": captions[index], "label": 0}

def mismatched_pair_map(example, idx, captions):
    """
    Creates mismatched pairs using the map function and indices.
    Args:
        example (HuggingFace Dataset): A single data example.
        idx (int): Index for the current example in the dataset.
        captions (list): A list of captions to select mismatched ones from.
    Returns:
        dict: A mismatched image-caption pair with a label (0).
    """
    return create_mismatched_pair(example, idx, captions)

def main():
    mscoco = load_dataset("clip-benchmark/wds_mscoco_captions")

    mscoco_train = mscoco["train"]
    mscoco_test = mscoco["test"]

    mscoco_train = mscoco_train.map(extract_first_caption)
    mscoco_test = mscoco_test.map(extract_first_caption)

    shuffled_dataset = mscoco_train.shuffle(seed=42)
    separation_point = len(shuffled_dataset) // 2

    shuffled_test = mscoco_test.shuffle(seed=42)
    t_separation_point = len(shuffled_test) // 2

    matching_pairs = shuffled_dataset.select(range(separation_point)).map(create_matching_pair)
    matching_pairs_ts = shuffled_test.select(range(t_separation_point)).map(create_matching_pair)

    captions = list(mscoco_train["caption"])
    random.shuffle(captions)

    captions_ts = list(mscoco_test["caption"])
    random.shuffle(captions_ts)

    mismatched_pairs = shuffled_dataset.select(range(separation_point, len(shuffled_dataset))).map(
        partial(mismatched_pair_map, captions=captions),  
        with_indices=True,
    )

    mismatched_pairs_ts = shuffled_test.select(range(t_separation_point, len(shuffled_test))).map(
        partial(mismatched_pair_map, captions=captions_ts),  
        with_indices=True,
    )

    combined_data = {
        "image": matching_pairs["image"] + mismatched_pairs["image"],
        "caption": matching_pairs["caption"] + mismatched_pairs["caption"],
        "label": matching_pairs["label"] + mismatched_pairs["label"],
    }

    combined_data_ts = {
        "image": matching_pairs_ts["image"] + mismatched_pairs_ts["image"],
        "caption": matching_pairs_ts["caption"] + mismatched_pairs_ts["caption"],
        "label": matching_pairs_ts["label"] + mismatched_pairs_ts["label"],
    }

    image_text_matching_ds = Dataset.from_dict(combined_data)
    image_txt_matching_test = Dataset.from_dict(combined_data_ts)

    image_text_matching_ds = image_text_matching_ds.shuffle(seed=42)
    image_txt_matching_test = image_txt_matching_test.shuffle(seed=42)

    image_text_matching_ds.save_to_disk("image_text_dataset")
    image_txt_matching_test.save_to_disk("image_text_test")


if __name__ == "__main__":
    main()
