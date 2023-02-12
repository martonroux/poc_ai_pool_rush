from model import App

app = App(lr=0.00001, data_path='data_2/emails.csv', data_batch_size=64, epoch=120)
app.run()
app.test_own_data('test_file.txt')
