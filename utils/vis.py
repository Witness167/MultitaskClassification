import matplotlib.pyplot as plt
import random


def plot_image(dataset, choice=None):
    """
    Visual image and label
    """
    if choice:
        idx = int(choice)
    else:
        idx = random.randint(0, len(dataset))

    image = dataset[idx][0]
    face = int(dataset[idx][1])
    mouth = int(dataset[idx][2])
    eyebrow = int(dataset[idx][3])
    eye = int(dataset[idx][4])
    nose = int(dataset[idx][5])
    jaw = int(dataset[idx][6])

    plt.imshow(image)
    plt.title(f"Face:{face} - Mouth:{mouth} - EyeBrow:{eyebrow} - Eye:{eye} - Nose:{nose} - Jaw:{jaw}")
    plt.show()
