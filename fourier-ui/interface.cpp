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

void MyForm::CalculateFourier() { // function to perform fourier calculations by accessing the functions in fourier-lib
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

	fourier->SelectDevice(gpu_select->SelectedIndex);  
	Params input;
	input.numHarmonics = Ng;

	Result output = fourier->Calculate(input, x_values, y_values); 
	label15->Text = Convert::ToString(output.executionTimeMs) + " ms"; // show a timer
	if (output.isSuccess) { DrawGraphics(output, x_values, y_values); } // draw charts if successfull
	else { MessageBox::Show("Error occured! : " + gcnew String(output.errorMessage.c_str()), "Error");}; // show an error if unsuccessful
}

void MyForm::DrawGraphics(const Result& result, const std::vector<double>& x_values, const std::vector<double>& y_values) { // draw graphics from calculated data
	Pen^ pen1 = gcnew Pen(Color::Blue, (float)(numericUpDown1->Value)); // chart color
	Pen^ pen2 = gcnew Pen(Color::Red, (float)(numericUpDown2->Value)); // x and y axes color
	Pen^ pen3 = gcnew Pen(Color::Green, (float)(numericUpDown3->Value)); // grid color
	Pen^ pen4 = gcnew Pen(Color::HotPink, (float)(numericUpDown4->Value)); // fourier series color
	Graphics^ g = pictureBox1->CreateGraphics();
	g->Clear(Color::White); // clear the screen
	int pb_Height = pictureBox1->Height; // get the height and width of pictureBox1
	int pb_Width = pictureBox1->Width; 
	int L = 40;
	const size_t Ne = x_values.size(); // setting the nunber of dots

	double minx = x_values.front(); // calculating minimum and maximum
	double maxx = x_values.back();

	double miny = y_values[0]; // initializing the variables
	double maxy = y_values[0];
	double minYg = result.calculatedY[0];
	double maxYg = result.calculatedY[0];

	for (size_t i = 1; i < Ne; ++i) { // calculating the extremums by iterating arrays
		if (y_values[i] > maxy) maxy = y_values[i];
		if (y_values[i] < miny) miny = y_values[i];
		if (result.calculatedY[i] > maxYg) maxYg = result.calculatedY[i];
		if (result.calculatedY[i] < minYg) minYg = result.calculatedY[i];
	}

	if (miny > minYg) { miny = minYg; }
	if (maxy < maxYg) { maxy = maxYg; }
	if (maxx == minx || maxy == miny) return; // additional check to prevent dividing by zero

	double Kx = (pb_Width - 2 * L) / (maxx - minx);
	double Ky = (pb_Height - 2 * L) / (miny - maxy);
	double Zx = (pb_Width * minx - L * (maxx + minx)) / (minx - maxx);
	double Zy = (pb_Height * maxy - L * (miny + maxy)) / (maxy - miny);
	double Gx, Gy;
	if (minx * maxx <= 0.0) Gx = 0.0;
	if (minx * maxx > 0.0) Gx = minx;
	if (minx * maxx > 0.0 && minx < 0.0) Gx = maxx;
	if (miny * maxy <= 0.0) Gy = 0.0;
	if (miny * maxy > 0.0 && miny > 0.0) Gy = miny;
	if (miny * maxy > 0.0 && miny < 0.0) Gy = maxy;
	int KrokX = (pb_Width - 2 * L) / 10;
	int KrokY = (pb_Height - 2 * L) / 9;
	for (int i = 1; i < 7; i++)
	{
		g->DrawLine(pen3, L, Math::Round(Ky * Gy + Zy, 4) + i * KrokY, pb_Width - L,
			Math::Round(Ky * Gy + Zy, 4) + i * KrokY);
		g->DrawLine(pen3, L, Math::Round(Ky * Gy + Zy, 4) - i * KrokY, pb_Width - L,
			Math::Round(Ky * Gy + Zy, 4) - i * KrokY);
	}
	for (int i = 1; i < 9; i++)
	{
		g->DrawLine(pen3, Math::Round(Kx * Gx + Zx, 4) + i * KrokX, L - 20,
			Math::Round(Kx * Gx + Zx, 4) + i * KrokX, pb_Height - L + 30);
		g->DrawLine(pen3, Math::Round(Kx * Gx + Zx, 4) - i * KrokX, L - 20,
			Math::Round(Kx * Gx + Zx, 4) - i * KrokX, pb_Height - L + 30);
	}
	
	double xx = minx; 
	double yy = maxy;
	double krx = (maxx - minx) / 10.0;
	double kry = (maxy - miny) / 10.6;
	for (int i = 0; i < 12; i++)
	{
		g->DrawString(Convert::ToString(Math::Round(xx, 1)), gcnew Drawing::Font("Times", 8),
			Brushes::Black, L + 4 + i * KrokX, Math::Round(Ky * Gy + Zy, 4) - L + 40.0f);
		g->DrawString(Convert::ToString(Math::Round(yy, 1)), gcnew Drawing::Font("Times", 8),
			Brushes::Black, Math::Round(Kx * Gx + Zx, 4) - L + 10.0f, (float)(L + i * KrokY) - 24.0f);
		xx = xx + krx;
		yy = yy - kry;
	}
	g->DrawLine(pen2, L, Math::Round(Ky * Gy + Zy, 4), Math::Round(pb_Width - 10, 4),
		Math::Round(Ky * Gy + Zy, 4)); // axis y
	g->DrawLine(pen2, Math::Round(Kx * Gx + Zx, 4), 10, Math::Round(Kx * Gx + Zx, 4),
		Math::Round(pb_Height - 10, 4)); // axis x
	for (int i = 1; i <= Ne - 1; i++)
	{
		g->DrawLine(pen1, Math::Round(Kx * x_values[i - 1] + Zx, 4), Math::Round(Ky * y_values[i - 1] + Zy, 4),
			Math::Round(Kx * x_values[i] + Zx, 4), Convert::ToInt32(Math::Round(Ky * y_values[i] + Zy, 4)));
	}
	minYg = result.calculatedY[0]; maxYg = result.calculatedY[0];
	for (int i = 1; i <= Ne - 1; i++)
	{
		if (minYg > result.calculatedY[i]) minYg = result.calculatedY[i];
	}
	for (int i = 1; i <= Ne - 1; i++)
	{
		g->DrawLine(pen4, Math::Round(Kx * x_values[i - 1] + Zx, 4), Math::Round(Ky * result.calculatedY[i - 1] + Zy, 4),
			Math::Round(Kx * x_values[i] + Zx, 4), Convert::ToInt32(Math::Round(Ky * result.calculatedY[i] + Zy, 4)));
	}
}