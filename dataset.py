#%%
import argparse
import utils as ut

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--path',metavar = 'path', help = "Path to the folder containing grain-free images", type = str, required=True)
    args = parser.parse_args()
    data_path = args.path

    answer = ''
    while answer.__class__ != bool:
        answer = input('Is your dataset made of grayscale .png images? yes/no: ')

        if answer.lower() == 'yes':
            answer = True
        elif answer.lower() == 'no':
            answer = False
        else:
            print('Type yes/no')

    if not answer:
        print('We are going to create a new one with your images converted.'+'\n')
        data_path = input('Please indicate the path where you want to save this new dataset (Press "Enter" to use '+args.path+'_gray)'+'\n'+':')

        while data_path == args.path:
            print('\nError! You indicated the same path as the one where your images are stored, you must provide a different one.')
            print('-'*10)
            data_path = input('Please indicate the path where you want to save this new dataset (Press "Enter" to use '+args.path+'_gray)'+'\n'+':')
        
        if data_path == '':
            if args.path[-1]=='/':
               data_path = args.path[:-1]+'_gray'
            else:
                data_path = args.path+'_gray'

        rdc = ut.RGB2Gray_Dataset_Convertor(args.path)
        rdc(save_folder=data_path)
    
    print('-'*50+'\n')
    final_path = input('Please indicate the path to the folder where you want to save your grainy dataset (Press "Enter" to use '+data_path+'_grain)'+'\n'+':')

    while (final_path == args.path) or (final_path == data_path):
            print('\nError! You indicated the same path as the one where your images are stored, you must provide a different one.')
            print('-'*10)
            final_path = input('Please indicate the path to the folder where you want to save your grainy dataset (Press "Enter" to use '+data_path+'_grain)'+'\n'+':')

    if final_path == '':
        if args.path[-1]=='/':
           final_path = args.path[:-1]+'_gray'
        else:
            final_path = args.path+'_gray'

    newson_path = input('Please indicate the path to the executable file of Newson etal. code'+'\n'+':')
    
    sure = False
    while (not sure) & (newson_path[-30:]!='/bin/film_grain_rendering_main'):
        print("\nWarning! The end of the path registered is different from '/bin/film_grain_rendering_main' (this is what is expected by default)")
        print('-'*10)
        answer2 = ''
        while answer2.__class__ != bool:
            answer2 = input('Are you sure of your path? yes/no: ')
            if answer2.lower() == 'yes':
                answer2 = True
            elif answer2.lower() == 'no':
                answer2 = False
            else:
                print('Type yes/no')
        if answer2:
            break
        newson_path = input('Please indicate the path to the executable file of Newson etal. code'+'\n'+':')
    print('-'*50+'\n')
    gdg = ut.GrainDatasetGenerator(data_path, newson_path)
    gdg(save_folder=final_path)
    print('Done!')
# %%
