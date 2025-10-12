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
	Pen^ pen1 = gcnew Pen(Color::Blue, (float)(numericUpDown1->Value)); // ���� �������
	Pen^ pen2 = gcnew Pen(Color::Red, (float)(numericUpDown2->Value)); // ���� ���� ���������
	Pen^ pen3 = gcnew Pen(Color::Green, (float)(numericUpDown3->Value)); // ���� ������
	Pen^ pen4 = gcnew Pen(Color::HotPink, (float)(numericUpDown4->Value)); // ���� ���� ����
	Graphics^ g = pictureBox1->CreateGraphics(); // ��������� �ᒺ��� g ��� ������ � ��������
	g->Clear(Color::White); // �������� �ᒺ��� g ��� ������ � ��������
	int pb_Height = pictureBox1->Height; // ������ � ������� ���������� pictureBox1
	int pb_Width = pictureBox1->Width; // ������ � ������� ���������� pictureBox1
	L = 40;
}