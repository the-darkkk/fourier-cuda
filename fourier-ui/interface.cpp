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
	float executionTimeMs = 0.0f;
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
	if (isSuccess) { DrawGraphics(calculatedY, x_values, y_values); } // draw charts if successfull
	else { MessageBox::Show("Error occured! : " + gcnew String(errorMessage.c_str()), "Error");}; // show an error if unsuccessful
}

void MyForm::DrawGraphics(const std::vector<double>& calculatedY, const std::vector<double>& x_values, const std::vector<double>& y_values) {
	this->chart1->Series->SuspendUpdates();
	this->chart1->Series->Clear();

	Series^ sOriginal = gcnew Series("Original function");
	sOriginal->ChartType = SeriesChartType::FastLine; 
	sOriginal->Color = Color::Blue;
	sOriginal->BorderWidth = (int)numericUpDown1->Value;

	Series^ sFourier = gcnew Series("Fourier Series");
	sFourier->ChartType = SeriesChartType::FastLine;
	sFourier->Color = Color::HotPink;
	sFourier->BorderWidth = (int)numericUpDown4->Value;

	size_t count = x_values.size();

	for (size_t i = 0; i < count; i++) {
		sOriginal->Points->AddXY(x_values[i], y_values[i]);

		if (i < calculatedY.size()) { 
			sFourier->Points->AddXY(x_values[i], calculatedY[i]);
		}
	}

	this->chart1->Series->Add(sOriginal);
	this->chart1->Series->Add(sFourier);
	this->chart1->Series->ResumeUpdates();
	this->chart1->ChartAreas[0]->RecalculateAxesScale();
}
