from vec_resampling import *
from file_management import *
from audio_fft import *
from sklearn.decomposition import PCA, KernelPCA, DictionaryLearning
from alive_progress import alive_bar, config_handler
config_handler.set_global(
    title_length=30,
    unknown='horizontal'
)

N = 28
TRAIN_LIMIT = 100
GEN_N = 512


sr, matrix = import_wav_dataset('training_data/kicks', max_len=2)

#pca = PCA(n_components=256)
pca1 = KernelPCA(n_components=N, kernel='poly', fit_inverse_transform=True)

(num, rho_matrix, phi_matrix) = row_wise_fft(matrix)

with alive_bar(title='reshaping'):
    ml_training_data = np.hstack((rho_matrix, phi_matrix))

with alive_bar(title='PCA1'):
    temp = pca1.fit_transform(ml_training_data)

"""
with alive_bar(title='PCA2'):
    temp = pca2.fit_transform(temp)

with alive_bar(title='inverse PCA2'):
    temp = pca2.inverse_transform(temp)
"""

with alive_bar(title='inverse PCA1'):
    temp = pca1.inverse_transform(temp[:TRAIN_LIMIT])

with alive_bar(title='reshaping'):
    rho_matrix, phi_matrix = np.hsplit(temp, 2)

ifft = row_wise_ifft(num, rho_matrix, phi_matrix)

export_wav_dataset((sr, ifft), 'generated_data/ai_drums', prefix='train_')

components = np.eye(N)

with alive_bar(title='extracting principal components'):
    temp = pca1.inverse_transform(components)

with alive_bar(title='reshaping'):
    rho_matrix, phi_matrix = np.hsplit(temp, 2)

ifft = row_wise_ifft(num, rho_matrix, phi_matrix)

export_wav_dataset((sr, ifft), 'generated_data/ai_drums', prefix='test_')

generated = np.random.rand(GEN_N, N)*2.0 - 1.0

with alive_bar(title='generating random kicks'):
    temp = pca1.inverse_transform(generated)

with alive_bar(title='reshaping'):
    rho_matrix, phi_matrix = np.hsplit(temp, 2)

ifft = row_wise_ifft(num, rho_matrix, phi_matrix)

export_wav_dataset((sr, ifft), 'generated_data/ai_drums', prefix='gen_')



