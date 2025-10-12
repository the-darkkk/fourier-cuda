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
	al = Convert::ToDouble(textBox1->Text);
	bl = Convert::ToDouble(textBox2->Text);
	Ne = Convert::ToInt32(textBox3->Text);
	Ng = Convert::ToInt32(textBox4->Text);
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
}