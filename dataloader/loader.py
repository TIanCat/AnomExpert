from dataloader.dataset import CaseFolderDataset, collate_cases_no_image_pad
from torch.utils.data import DataLoader



def get_dataloader(args):
    
    
    train_dataset = CaseFolderDataset(
                    root_dir= args.data_dir,
                    split = 'train',
                    max_images= args.max_images_per_case,
                    augment=True)

    val_dataset = CaseFolderDataset(
                    root_dir= args.data_dir,
                    split = 'val',
                    max_images= args.max_images_per_case,
                    augment=False)

    test_dataset = CaseFolderDataset(
                    root_dir= args.data_dir,
                    split = 'test',
                    max_images= args.max_images_per_case,
                    augment=False)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn = collate_cases_no_image_pad)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn = collate_cases_no_image_pad)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn = collate_cases_no_image_pad)
    
    return train_dataloader, val_dataloader, test_dataloader 



    
    
    train_dataset = CaseFolderDataset2(
                    root_dir= args.data_dir,
                    split = 'train',
                    max_images= args.max_images)

    val_dataset = CaseFolderDataset2(
                    root_dir= args.data_dir,
                    split = 'val',
                    max_images= args.max_images)

    test_dataset = CaseFolderDataset2(
                    root_dir= args.data_dir,
                    split = 'test',
                    max_images= args.max_images)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn = collate_cases_no_image_pad2)

    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn = collate_cases_no_image_pad2)

    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn = collate_cases_no_image_pad2)
    
    return train_dataloader, val_dataloader, test_dataloader 