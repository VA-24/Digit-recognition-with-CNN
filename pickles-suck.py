import pickle

pickle_in = open("mnistdataset.pickle", "rb")
model = pickle.load(pickle_in)

#execute the following lines of code in order to see statistics of saved mdoel, commented out for now:

"""

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test loss is " + str(test_loss))
print("Test acc is " + str(test_acc * 100))
prediction = model.predict(X_test)
"""
