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
	double al = Convert::ToDouble(textBox1->Text);
	double bl = Convert::ToDouble(textBox2->Text);
	double Ne = Convert::ToInt32(textBox3->Text);
	double Ng = Convert::ToInt32(textBox4->Text);

	std::vector<double> x_values(Ne);
	std::vector<double> y_values(Ne);

	double h = (bl - al) / (Ne > 1 ? Ne - 1 : 1);
	for (int i = 0; i < Ne; ++i) {
		x_values[i] = al + i * h;
		switch (func_select->SelectedIndex) {
		case 0: y_values[i] = sin(x_values[i]); break;
	    // ..
		}
	}

	fourier->SelectDevice(0);
	Params input;
	input.numHarmonics = Ng;

	Result output = fourier->Calculate(input, x_values, y_values);
}

void MyForm::DrawGraphics(const Result& result, const std::vector<double>& x_values, const std::vector<double>& y_values) { // draw graphics from calculated data
	Pen^ pen1 = gcnew Pen(Color::Blue, (float)(numericUpDown1->Value)); // колір графіка
	Pen^ pen2 = gcnew Pen(Color::Red, (float)(numericUpDown2->Value)); // колір осей координат
	Pen^ pen3 = gcnew Pen(Color::Green, (float)(numericUpDown3->Value)); // колір ґратки
	Pen^ pen4 = gcnew Pen(Color::HotPink, (float)(numericUpDown4->Value)); // колір суми ряду
	Graphics^ g = pictureBox1->CreateGraphics(); // створення об’єкта g для роботи з графікою
	g->Clear(Color::White); // очищення об’єкта g для роботи з графікою
	int pb_Height = pictureBox1->Height; // висота в пікселях компоненти pictureBox1
	int pb_Width = pictureBox1->Width; // ширина в пікселях компоненти pictureBox1
	L = 40;

	minx = Xe[0]; maxx = Xe[Ne - 1];
	miny = Ye[0]; maxy = Ye[0];
	minYg = Yg[0]; maxYg = Yg[0];
	for (int i = 1; i <= Ne - 1; i++)
	{
		if (minYg > Yg[i]) minYg = Yg[i];
	}
	for (int i = 1; i <= Ne - 1; i++)
	{
		if (maxy < Ye[i]) maxy = Ye[i];
		if (miny > Ye[i]) miny = Ye[i];
	}
	if (miny > minYg) { miny = minYg; }
	if (maxy < maxYg) { maxy = maxYg; }
	Kx = (pb_Width - 2 * L) / (maxx - minx);
	Ky = (pb_Height - 2 * L) / (miny - maxy);
	Zx = (pb_Width * minx - L * (maxx + minx)) / (minx - maxx);
	Zy = (pb_Height * maxy - L * (miny + maxy)) / (maxy - miny);
	if (minx * maxx <= 0.0) Gx = 0.0;
	if (minx * maxx > 0.0) Gx = minx;
	if (minx * maxx > 0.0 && minx < 0.0) Gx = maxx;
	if (miny * maxy <= 0.0) Gy = 0.0;
	if (miny * maxy > 0.0 && miny > 0.0) Gy = miny;
	if (miny * maxy > 0.0 && miny < 0.0) Gy = maxy;
	KrokX = (pb_Width - 2 * L) / 10;
	KrokY = (pb_Height - 2 * L) / 9;
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
	}xx = minx; yy = maxy;
	krx = (maxx - minx) / 10.0;
	kry = (maxy - miny) / 10.6;
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
		Math::Round(Ky * Gy + Zy, 4)); // вісь У
	g->DrawLine(pen2, Math::Round(Kx * Gx + Zx, 4), 10, Math::Round(Kx * Gx + Zx, 4),
		Math::Round(pb_Height - 10, 4)); // вісь x
	for (int i = 1; i <= Ne - 1; i++)
	{
		g->DrawLine(pen1, Math::Round(Kx * Xe[i - 1] + Zx, 4), Math::Round(Ky * Ye[i - 1] + Zy, 4),
			Math::Round(Kx * Xe[i] + Zx, 4), Convert::ToInt32(Math::Round(Ky * Ye[i] + Zy, 4)));
	}
	minYg = Yg[0]; maxYg = Yg[0];
	for (int i = 1; i <= Ne - 1; i++)
	{
		if (minYg > Yg[i]) minYg = Yg[i];
	}
	for (int i = 1; i <= Ne - 1; i++)
	{
		g->DrawLine(pen4, Math::Round(Kx * Xe[i - 1] + Zx, 4), Math::Round(Ky * Yg[i - 1] + Zy, 4),
			Math::Round(Kx * Xe[i] + Zx, 4), Convert::ToInt32(Math::Round(Ky * Yg[i] + Zy, 4)));
	}
}