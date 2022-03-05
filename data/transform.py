from torchvision import transforms as T

to_tensor = T.Compose([
    T.ToTensor(),
    T.Resize((64, 64)),
    T.Normalize(mean=[0.9, 0.9, 0.9], std=[0.1, 0.1, 0.1]),
])

color = T.RandomApply(T.ColorJitter(0, 0, 0, 0), 0.2)

augmenter = T.Compose([
    color,
    T.RandomGrayscale(0.2),
    to_tensor
])