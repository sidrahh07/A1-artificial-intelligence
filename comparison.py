import matplotlib.pyplot as plt

# comparing accuracies
models = ['basic AI', 'advanced AI']
accuracy = [0.806, 0.7485]

plt.bar(models, accuracy)
plt.show()

# F1-score comparison
f1_false = [0.85, 0.73]
f1_true = [0.72, 0.76]

x = range(len(models))
plt.bar(x, f1_false, width=0.4, label='False')
plt.bar([i + 0.4 for i in x], f1_true, width=0.4, label='True')
plt.xticks([i + 0.2 for i in x], models)
plt.legend()
plt.show()
