from model import App

app = App(lr=0.001, data_path='data_2/emails.csv', data_batch_size=16, epoch=10)
model = app.run()
app.test_own_data('test_file.txt')
