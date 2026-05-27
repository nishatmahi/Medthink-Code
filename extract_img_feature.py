import torch
from PIL import Image
import torchvision.transforms as T
import os
import argparse
import json

def get_model(_img_type):
    print(f"Loading cooelf/detr Model from GitHub...")
    # This version matches the expected list/tuple output format
    _model = torch.hub.load('cooelf/detr:main', 'detr_resnet101_dc5', pretrained=True)
    _transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return _model, _transform

def extract_features(_model, _transform, _input_image, _img_type, _device):
    print(f"Loading {_input_image}...")
    img = Image.open(_input_image).convert("RGB")
    inp = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        return _model(inp)[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--image_dir",  type=str, required=True,
                        help="Directory containing the images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write detr.pth and name_map.json")
    parser.add_argument("--dataset",    type=str, choices=["rad", "slake", "path"], required=True,
                        help="rad: flat .jpg dir | slake: xmlab<n>/source.jpg subdirs | path: flat .jpg dir")
    parser.add_argument("--img_type",   type=str, default="detr")
    args = vars(parser.parse_args())
    for arg, value in args.items():
        print(f"{arg}: {value}")

    device     = args["device"] if torch.cuda.is_available() else "cpu"
    img_type   = args["img_type"]
    images_dir = args["image_dir"]
    output_dir = args["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    model, transform = get_model(img_type)
    model.to(device)
    model.eval()

    name_map        = {}   # KEY = image ID string, VALUE = index in feature matrix
    vision_features = []
    cnt             = 0

    # ── RAD: flat directory of synpic*.jpg files ──────────────────────────────
    if args["dataset"] == "rad":
        images_path = sorted(os.listdir(images_dir))
        print(f"There are {len(images_path)} entries in {images_dir}.")
        for image_path in images_path:
            if image_path.lower().endswith(".jpg"):
                full_path = os.path.join(images_dir, image_path)
                feature   = extract_features(model, transform, full_path, img_type, device)
                image_id  = image_path[:-4]          # e.g. "synpic54610"
                name_map[image_id] = str(cnt)
                vision_features.append(feature.detach().cpu())
                cnt += 1

    # ── SLAKE: subdirectory-per-image layout  xmlab<n>/source.jpg ─────────────
    # ✅ Fixed: original code incorrectly checked image_path.endswith('.jpg')
    #    but entries are directories (xmlab1, xmlab2 ...), not .jpg files.
    elif args["dataset"] == "slake":
        images_path = sorted(os.listdir(images_dir))
        print(f"There are {len(images_path)} entries in {images_dir}.")
        for image_path in images_path:
            subdir     = os.path.join(images_dir, image_path)
            source_img = os.path.join(subdir, "source.jpg")
            if os.path.isdir(subdir) and os.path.exists(source_img):
                feature  = extract_features(model, transform, source_img, img_type, device)
                # image_path is like "xmlab1" → strip "xmlab" prefix (5 chars) → "1"
                image_id = image_path[5:]              # e.g. "1", "100"
                name_map[image_id] = str(cnt)
                vision_features.append(feature.detach().cpu())
                cnt += 1

    # ── PathVQA: flat directory of train_*.jpg / test_*.jpg / val_*.jpg ───────
    # ✅ New: same logic as RAD — strip .jpg to get the image key
    elif args["dataset"] == "path":
        images_path = sorted(os.listdir(images_dir))
        print(f"There are {len(images_path)} entries in {images_dir}.")
        for image_path in images_path:
            if image_path.lower().endswith(".jpg"):
                full_path = os.path.join(images_dir, image_path)
                feature   = extract_features(model, transform, full_path, img_type, device)
                image_id  = image_path[:-4]            # e.g. "train_0422"
                name_map[image_id] = str(cnt)
                vision_features.append(feature.detach().cpu())
                cnt += 1

    if cnt == 0:
        raise RuntimeError(
            f"No images were processed from '{images_dir}'. "
            "Check that the image_dir path is correct and contains the expected files."
        )

    vision_features = torch.cat(vision_features)
    print(f"Feature matrix shape: {vision_features.shape}  ({cnt} images)")

    out_pth  = os.path.join(output_dir, f"{img_type}.pth")
    out_json = os.path.join(output_dir, "name_map.json")

    torch.save(vision_features, out_pth)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=4, ensure_ascii=False)

    print(f"Saved feature matrix → {out_pth}")
    print(f"Saved name map       → {out_json}")
    print("Done.")
