import json

def main():
    json_path = "../../data/nerf_voxel/our_desk_2_raw/transforms_full.json"
    json_path_save = "../../data/nerf_voxel/our_desk_2_raw/transforms_full_new.json"
    f = open(json_path,"r")
    meta = json.load(f)
    f.close()

    for frame in meta["frames"]:
        frame["file_path"] = frame["file_path"].lstrip("./full/")

    num = len(meta['frames'])
    print(f'find {num} examples')
    f = open(json_path_save, "w")
    json.dump(meta, f, indent=6)
    f.close()

if __name__ == "__main__":
    main()