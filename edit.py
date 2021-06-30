import matplotlib.pyplot as plt

with open("ELM MAPE Dsb.txt", "r") as f:
	data = f.readlines()
with open("ANN MAPE Dsb.txt", "r") as f:
	data2 = f.readlines();

v_actual = []
t_actual = []
v_elm = []
t_elm = []
v_ann = []
t_ann = []

for i in range(51):
	if(i > 11):
		tmp1 = data[i].rstrip().split("\t")
		tmp1_int = [float(i) for i in tmp1 if i != '']
		tmp2 = data2[i].rstrip().split("\t")
		tmp2_int = [float(i) for i in tmp2 if i != '']
		v_actual.append(tmp1_int[0])
		t_actual.append(tmp1_int[1])
		v_elm.append(tmp1_int[2])
		t_elm.append(tmp1_int[3])
		v_ann.append(tmp2_int[2])
		t_ann.append(tmp2_int[3])


plt.plot(t_actual, label='Actual')
plt.plot(t_elm, label='ELM')
plt.plot(t_ann, label='ANN')
plt.title("Phase Angle Actual vs ELM Method vs ANN Method\n", fontweight='bold')
plt.ylabel(r"Phase Angle ($\dot{\Theta}$)")
plt.xlabel("N Data")
plt.legend()
plt.show()
