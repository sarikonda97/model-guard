import model
import modes

def main():
	oracle_path = "C:/Users/jashw/Desktop/Integration/Attack Detection/src/cifar_net.pt"
	modes.serve_model(0.9, oracle_path, model.YourModel)

if __name__ == "__main__":
	main()
