#include "interface.h"
#include <Windows.h>
#include <tchar.h>

using namespace fourierui;
[STAThread]
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	Application::Run(gcnew MyForm);
	return 0;
}

void MyForm::CalculateFourier() { // function to perform fourier calculations by accessing the functions in fourier-lib or fourier-cpu-lib
	double al, bl;
	unsigned int Ne, Ng;
	try {
		al = Convert::ToDouble(textBox1->Text);
		bl = Convert::ToDouble(textBox2->Text);
		Ng = Convert::ToUInt32(textBox4->Text);
		if (cb_auto->Checked) { Ne = 3 * Ng; textBox3->Text = Convert::ToString(Ne); } // automatic way - usually 3 points is enough to have a good-looking garmonics
		else { Ne = Convert::ToUInt32(textBox3->Text); };
		if (Ng == 0 || Ne == 0) { throw -1; };
	}
	catch (...) {
		MessageBox::Show("Bad input parameters!", "Error"); return;
	};

	std::vector<double> x_values(Ne);
	std::vector<double> y_values(Ne);

	double h = (bl - al) / (Ne > 1 ? Ne - 1 : 1);
	for (int i = 0; i < Ne; ++i) {
		x_values[i] = al + i * h;
		switch (func_select->SelectedIndex) {
		case 0: y_values[i] = sin(x_values[i]); break;
		case 1: y_values[i] = abs(((int)x_values[i] % 2) - 1); break;
		case 2: y_values[i] = sin(x_values[i]) + cos(2 * x_values[i]); break;
		case 3: y_values[i] = abs(cos(x_values[i])); break;
		default: MessageBox::Show("No function selected!", "Error"); return;
		}
	}

	std::vector<double> calculatedY;
	double executionTimeMs = 0.0f;
	bool isSuccess = false;
	std::string errorMessage = "";

	if (cb_cpu->Checked) {
		FourierCPU::FourierCpuCalculator calc;
		FourierCPU::Params p;
		p.numHarmonics = Ng;
		// skip the device select
		auto res = calc.Calculate(p, x_values, y_values); // CPU Calculation

		// save the data to variables
		calculatedY = res.calculatedY;
		executionTimeMs = res.executionTimeMs;
		isSuccess = res.isSuccess;
		errorMessage = res.errorMessage;
	}
	else {
		FourierGPU::FourierCudaCalculator calc;
		FourierGPU::Params p;
		p.numHarmonics = Ng;
		calc.SelectDevice(gpu_select->SelectedIndex);
		auto res = calc.Calculate(p, x_values, y_values); // GPU Calculation

		// save the data to variables
		calculatedY = res.calculatedY;
		executionTimeMs = res.executionTimeMs;
		isSuccess = res.isSuccess;
		errorMessage = res.errorMessage;
	}

	label15->Text = Convert::ToString(executionTimeMs) + " ms"; // show a timer
	if (isSuccess) { DrawGraphics(x_values, y_values, calculatedY, false, "Initial function", "Fourier series"); } // draw charts if successfull
	else { MessageBox::Show("Error occured! : " + gcnew String(errorMessage.c_str()), "Error");}; // show an error if unsuccessful
}

void MyForm::DrawGraphics(const std::vector<double>& x_values, const std::vector<double>& y_values, const std::vector<double>& y2_values, bool IsLogarithmic, const char* name1, const char* name2) {
	this->chart1->Series->SuspendUpdates();
	this->chart1->Series->Clear();

	if (IsLogarithmic) {
		this->chart1->ChartAreas[0]->AxisY->IsLogarithmic = true;
		this->chart1->ChartAreas[0]->AxisX->IsLogarithmic = true;
	}
	else {
		this->chart1->ChartAreas[0]->AxisY->IsLogarithmic = false;
		this->chart1->ChartAreas[0]->AxisX->IsLogarithmic = false;
	}

	Series^ sOriginal = gcnew Series(gcnew String(name1));
	sOriginal->ChartType = SeriesChartType::FastLine; 
	sOriginal->Color = Color::Blue;
	sOriginal->BorderWidth = (int)numericUpDown1->Value;

	Series^ sFourier = gcnew Series(gcnew String(name2));
	sFourier->ChartType = SeriesChartType::FastLine;
	sFourier->Color = Color::HotPink;
	sFourier->BorderWidth = (int)numericUpDown4->Value;

	size_t count = x_values.size();

	for (size_t i = 0; i < count; i++) {
		sOriginal->Points->AddXY(x_values[i], y_values[i]);

		if (i < y2_values.size()) { 
			sFourier->Points->AddXY(x_values[i], y2_values[i]);
		}
	}

	this->chart1->Series->Add(sOriginal);
	this->chart1->Series->Add(sFourier);
	this->chart1->Series->ResumeUpdates();
	this->chart1->ChartAreas[0]->RecalculateAxesScale();
}

void MyForm::PerformTest() {
	if (gpu_select->SelectedIndex == -1) {
		MessageBox::Show("Select the GPU!", "Error!");
		return;
	};
	button4->Text = "Виконується...";
	
	double al = -3.14159265358979323846;
	double bl = 3.14159265358979323846; // limits
	int test_count = 10; // steps

	std::vector<double> N_squared_values;   // x-vals (Complexity)
	std::vector<double> gpu_time_values;    // y-vals gpu
	std::vector<double> cpu_time_values;    // y-vals cpu
	N_squared_values.reserve(test_count);
	gpu_time_values.reserve(test_count);
	cpu_time_values.reserve(test_count);

	{ // a warmup test for gpu - just calculates something and does not save
		FourierGPU::FourierCudaCalculator warmUpCalc;
		FourierGPU::Params p; p.numHarmonics = 10;
		std::vector<double> dummy(10, 0.0);
		warmUpCalc.SelectDevice(gpu_select->SelectedIndex);
		warmUpCalc.Calculate(p, dummy, dummy);
	}

	unsigned int N = 64;
	for (int j = 0; j < test_count; j++) { // tabulate the input data as sin(x) function
		std::vector<double> x_values(N);
		std::vector<double> y_values(N); 
		
		double h = (bl - al) / (double)(N > 1 ? N - 1 : 1);
		for (unsigned int i = 0; i < N; ++i) {
			x_values[i] = al + i * h;
			y_values[i] = sin(x_values[i]);
		};

		// testing gpu
		FourierGPU::FourierCudaCalculator calc_gpu;
		FourierGPU::Params p_gpu;
		p_gpu.numHarmonics = N; // because alghorythm has O(N^2) complexity
		calc_gpu.SelectDevice(gpu_select->SelectedIndex);

		auto res_gpu = calc_gpu.Calculate(p_gpu, x_values, y_values);

		// the same for cpu
		FourierCPU::FourierCpuCalculator calc_cpu;
		FourierCPU::Params p_cpu;
		p_cpu.numHarmonics = N;
		auto res_cpu = calc_cpu.Calculate(p_cpu, x_values, y_values);

		// vector of complexity
		double complexity = (double)N * (double)N;
		N_squared_values.push_back(complexity);
		
		// check if the time is below zero
		if (res_gpu.executionTimeMs <= 0.0) { res_gpu.executionTimeMs = 1e-3; };
		if (res_cpu.executionTimeMs <= 0.0) { res_cpu.executionTimeMs = 1e-3; };
		gpu_time_values.push_back(res_gpu.executionTimeMs);
		cpu_time_values.push_back(res_cpu.executionTimeMs);
		N *= 2; // double the N
	}
	button4->Text = "Тестування";
	DrawGraphics(N_squared_values, cpu_time_values, gpu_time_values, true, "CPU", "GPU");
}