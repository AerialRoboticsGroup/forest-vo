from os import system, mkdir
import argparse
from os.path import isdir, isfile, join
import boto3
from botocore.exceptions import NoCredentialsError
from gluefactory.settings import DATA_PATH
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='TartanAir')

    parser.add_argument('--output-dir', default='./',
                        help='root directory for downloaded files')

    parser.add_argument('--rgb', action='store_true', default=False,
                        help='download rgb image')

    parser.add_argument('--depth', action='store_true', default=False,
                        help='download depth image')

    parser.add_argument('--flow', action='store_true', default=False,
                        help='download optical flow')

    parser.add_argument('--seg', action='store_true', default=False,
                        help='download segmentation image')

    parser.add_argument('--only-easy', action='store_true', default=False,
                        help='download only easy trajectories')

    parser.add_argument('--only-hard', action='store_true', default=False,
                        help='download only hard trajectories')

    parser.add_argument('--only-left', action='store_true', default=False,
                        help='download only left camera')

    parser.add_argument('--only-right', action='store_true', default=False,
                        help='download only right camera')

    parser.add_argument('--only-flow', action='store_true', default=False,
                        help='download only optical flow wo/ mask')

    parser.add_argument('--only-mask', action='store_true', default=False,
                        help='download only mask wo/ flow')

    # parser.add_argument('--azcopy', action='store_true', default=False,
    #                     help='download the data with AzCopy, which is 10x faster in our test')

    args = parser.parse_args()

    return args

def _help():
    print ('')

def download_from_cloudflare_r2(s3, filelist, destination_path, bucket_name):
    """
    Downloads a file from Cloudflare R2 storage using S3 API.

    Args:
    - file_name (str): Name of the file in the bucket you want to download
    - destination_path (str): Path to save the downloaded file locally
    - bucket_name (str): The name of the Cloudflare R2 bucket
    - access_key (str): Your Cloudflare R2 access key
    - secret_key (str): Your Cloudflare R2 secret key
    - endpoint_url (str): Endpoint URL for Cloudflare R2

    Returns:
    - str: A message indicating success or failure.
    """


    for file_name in filelist:
        target_file_name = join(destination_path, file_name.replace('/', '_').replace('tartanair_',''))
        print("/".join(target_file_name.split('/')[6:]))
        print('--')
        if isfile(target_file_name):
            print('Skipping: Target file {} already exists..'.format(target_file_name))
            continue
        try:
            print(f"  Downloading {file_name} from {bucket_name}...")
            s3.download_file(bucket_name, file_name, target_file_name)
            print(f"  Successfully downloaded {file_name} to {target_file_name}")
        except FileNotFoundError:
            print(f"Error: The file {file_name} was not found in the bucket {bucket_name}.")
        except NoCredentialsError:
            print("Error: Credentials not available.")

def get_all_s3_objects(s3, bucket_name):
    continuation_token = None
    content_list = []
    while True:
        list_kwargs = dict(MaxKeys=1000, Bucket = bucket_name)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        content_list.extend(response.get('Contents', []))
        if not response.get('IsTruncated'):  # At the end of the list?
            break
        continuation_token = response.get('NextContinuationToken')
    return content_list

def get_size(content_list, filelist):
    keys_sizes = {rrr['Key']: rrr['Size'] for rrr in content_list}
    total_size = 0
    for ff in filelist:
        total_size += keys_sizes[ff]
    return total_size

if __name__ == '__main__':
    bucket_name = "tartanair-v1"
    # If this fails go to: https://github.com/castacks/tartanair_tools/blob/master/download_training.py
    # and get the latest access key and secret key from there
    access_key = "f1ae9efebbc6a9a7cebbd949ba3a12de"
    secret_key = "0a21fe771089d82e048ed0a1dd6067cb29a5666bf4fe95f7be9ba6f72482ec8b"
    endpoint_url = "https://0a585e9484af268a716f8e6d3be53bbc.r2.cloudflarestorage.com"

    args = get_args()

    outdir = join(DATA_PATH, 'syntheticForestData')

    # difficulty level
    levellist = ['Easy', 'Hard']
    if args.only_easy:
        levellist = ['Easy']
    if args.only_hard:
        levellist = ['Hard']
    if args.only_easy and args.only_hard:
        print('--only-eazy and --only-hard tags can not be set at the same time!')
        exit()


    # filetype
    typelist = []
    if args.rgb:
        typelist.append('image')
    if args.depth:
        typelist.append('depth')
    if args.seg:
        typelist.append('seg')
    if args.flow:
        typelist.append('flow')
    if len(typelist)==0:
        print('Specify the type of data you want to download by --rgb/depth/seg/flow')
        exit()

    # camera 
    cameralist = ['left', 'right', 'flow', 'mask']
    if args.only_left:
        cameralist.remove('right')
    if args.only_right:
        cameralist.remove('left')
    if args.only_flow:
        cameralist.remove('mask')
    if args.only_mask:
        cameralist.remove('flow')
    if args.only_left and args.only_right:
        print('--only-left and --only-right tags can not be set at the same time!')
        exit()
    if args.only_flow and args.only_mask:
        print('--only-flow and --only-mask tags can not be set at the same time!')
        exit()

    # read all the zip file urls
    with open(f'{DATA_PATH}/tartanAirDownloader/download_training_zipfiles.txt') as f:
        lines = f.readlines()
    ziplist = [ll.strip() for ll in lines if ll.strip().endswith('.zip')]

    downloadlist = []
    for zipfile in ziplist:
        zf = zipfile.split('/')
        filename = zf[-1]
        difflevel = zf[-2]

        # image/depth/seg/flow
        filetype = filename.split('_')[0] 
        # left/right/flow/mask
        cameratype = filename.split('.')[0].split('_')[-1]
        
        if (difflevel in levellist) and (filetype in typelist) and (cameratype in cameralist):
            downloadlist.append(zipfile) 

    if len(downloadlist)==0:
        print('No file meets the condition!')
        exit()

    print('{} files are going to be downloaded...'.format(len(downloadlist)))
    for fileurl in downloadlist:
        print ('  -', fileurl)

    # Create an S3 client with the provided credentials and endpoint
    s3 = boto3.client('s3', aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key,
                      endpoint_url=endpoint_url)

    # Print out how much space
    content_list = get_all_s3_objects(s3, bucket_name)
    all_size = get_size(content_list, downloadlist)
    print('*** Total Size: {} GB ***'.format(all_size/1000000000))
    download_from_cloudflare_r2(s3, downloadlist, outdir, bucket_name)
    print('*** Download finished! ***')
    print('*** Please check the downloaded files in {} ***'.format(outdir))

