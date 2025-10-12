#pragma once 

#include <cmath>
#include "../fourier-lib/fourier-lib.h"
#pragma comment(lib, "fourier-lib.lib")

namespace fourierui {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	double Xe[1000]; // масив із координатами аргументів графіків
	double Ye[1000]; // масив із координатами точок періодичної функції
	double Yg[1000]; // масив із координатами суми ряду Фур’є
	double c[50]; // масив із значеннями амплітуд гармонік

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			fourier = new FourierCudaCalculator();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
			delete fourier;
		}
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: System::Windows::Forms::GroupBox^ groupBox1;

	private: System::Windows::Forms::Label^ label9;
	private: System::Windows::Forms::Label^ label8;
	private: System::Windows::Forms::Label^ label7;
	private: System::Windows::Forms::Label^ label6;
	private: System::Windows::Forms::Label^ label5;

	private: System::Windows::Forms::Label^ label3;

	private: System::Windows::Forms::Label^ label12;
	private: System::Windows::Forms::Label^ label11;
	private: System::Windows::Forms::Label^ label10;
	private: System::Windows::Forms::TextBox^ textBox3;
	private: System::Windows::Forms::TextBox^ textBox2;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown3;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown2;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown1;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::TextBox^ textBox4;
	private: System::Windows::Forms::Label^ label2;

	private: System::Windows::Forms::Label^ label14;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown4;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::ComboBox^ func_select;

	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::Label^ label13;
	private: System::Windows::Forms::ComboBox^ integral_select;

	protected:

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container^ components;

	private:
		FourierCudaCalculator* fourier; // create a pointer to fourier-calculation class
		void CalculateFourier();
		void DrawGraphics(const Result& result, const std::vector<double>& x_values, const std::vector<double>& y_values);

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->label13 = (gcnew System::Windows::Forms::Label());
			this->integral_select = (gcnew System::Windows::Forms::ComboBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->func_select = (gcnew System::Windows::Forms::ComboBox());
			this->numericUpDown4 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->textBox4 = (gcnew System::Windows::Forms::TextBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label14 = (gcnew System::Windows::Forms::Label());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->numericUpDown3 = (gcnew System::Windows::Forms::NumericUpDown());
			this->numericUpDown2 = (gcnew System::Windows::Forms::NumericUpDown());
			this->numericUpDown1 = (gcnew System::Windows::Forms::NumericUpDown());
			this->textBox3 = (gcnew System::Windows::Forms::TextBox());
			this->textBox2 = (gcnew System::Windows::Forms::TextBox());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->label12 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->label10 = (gcnew System::Windows::Forms::Label());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->groupBox1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown4))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown3))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->BeginInit();
			this->SuspendLayout();
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(2, 3);
			this->pictureBox1->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(1030, 423);
			this->pictureBox1->TabIndex = 0;
			this->pictureBox1->TabStop = false;
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->label13);
			this->groupBox1->Controls->Add(this->integral_select);
			this->groupBox1->Controls->Add(this->label4);
			this->groupBox1->Controls->Add(this->func_select);
			this->groupBox1->Controls->Add(this->numericUpDown4);
			this->groupBox1->Controls->Add(this->label1);
			this->groupBox1->Controls->Add(this->textBox4);
			this->groupBox1->Controls->Add(this->label2);
			this->groupBox1->Controls->Add(this->label14);
			this->groupBox1->Controls->Add(this->button2);
			this->groupBox1->Controls->Add(this->button1);
			this->groupBox1->Controls->Add(this->numericUpDown3);
			this->groupBox1->Controls->Add(this->numericUpDown2);
			this->groupBox1->Controls->Add(this->numericUpDown1);
			this->groupBox1->Controls->Add(this->textBox3);
			this->groupBox1->Controls->Add(this->textBox2);
			this->groupBox1->Controls->Add(this->textBox1);
			this->groupBox1->Controls->Add(this->label9);
			this->groupBox1->Controls->Add(this->label8);
			this->groupBox1->Controls->Add(this->label7);
			this->groupBox1->Controls->Add(this->label6);
			this->groupBox1->Controls->Add(this->label5);
			this->groupBox1->Controls->Add(this->label12);
			this->groupBox1->Controls->Add(this->label3);
			this->groupBox1->Controls->Add(this->label11);
			this->groupBox1->Controls->Add(this->label10);
			this->groupBox1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(254)));
			this->groupBox1->ForeColor = System::Drawing::Color::Black;
			this->groupBox1->Location = System::Drawing::Point(3, 432);
			this->groupBox1->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Padding = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->groupBox1->Size = System::Drawing::Size(1030, 180);
			this->groupBox1->TabIndex = 1;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Параметри графіку";
			// 
			// label13
			// 
			this->label13->AutoSize = true;
			this->label13->ForeColor = System::Drawing::Color::SeaGreen;
			this->label13->Location = System::Drawing::Point(316, 41);
			this->label13->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label13->Name = L"label13";
			this->label13->Size = System::Drawing::Size(264, 15);
			this->label13->TabIndex = 25;
			this->label13->Text = L"Виберіть метод обчислення інтегралу";
			// 
			// integral_select
			// 
			this->integral_select->FormattingEnabled = true;
			this->integral_select->Items->AddRange(gcnew cli::array< System::Object^  >(4) {
				L"Звичайний", L"Центр. Прям.", L"Трапецій",
					L"Сімпсона"
			});
			this->integral_select->Location = System::Drawing::Point(319, 60);
			this->integral_select->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->integral_select->Name = L"integral_select";
			this->integral_select->Size = System::Drawing::Size(150, 23);
			this->integral_select->TabIndex = 5;
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->ForeColor = System::Drawing::Color::SeaGreen;
			this->label4->Location = System::Drawing::Point(316, 116);
			this->label4->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(204, 15);
			this->label4->TabIndex = 23;
			this->label4->Text = L"Виберіть періодичну функцію";
			// 
			// func_select
			// 
			this->func_select->FormattingEnabled = true;
			this->func_select->Items->AddRange(gcnew cli::array< System::Object^  >(4) {
				L"sin(x)", L"f(x) = |(x mod 2) - 1|", L"f(x) = sin(x) + cos(2x)",
					L"f(x) = |cos(x)|"
			});
			this->func_select->Location = System::Drawing::Point(319, 136);
			this->func_select->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->func_select->Name = L"func_select";
			this->func_select->Size = System::Drawing::Size(150, 23);
			this->func_select->TabIndex = 6;
			// 
			// numericUpDown4
			// 
			this->numericUpDown4->Location = System::Drawing::Point(758, 124);
			this->numericUpDown4->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->numericUpDown4->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
			this->numericUpDown4->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDown4->Name = L"numericUpDown4";
			this->numericUpDown4->Size = System::Drawing::Size(37, 21);
			this->numericUpDown4->TabIndex = 22;
			this->numericUpDown4->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(566, 124);
			this->label1->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(178, 15);
			this->label1->TabIndex = 21;
			this->label1->Text = L"Товщина лінії суми ряду =";
			// 
			// textBox4
			// 
			this->textBox4->Location = System::Drawing::Point(75, 133);
			this->textBox4->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->textBox4->Name = L"textBox4";
			this->textBox4->Size = System::Drawing::Size(76, 21);
			this->textBox4->TabIndex = 4;
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->ForeColor = System::Drawing::Color::SeaGreen;
			this->label2->Location = System::Drawing::Point(26, 116);
			this->label2->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(188, 15);
			this->label2->TabIndex = 18;
			this->label2->Text = L"Введіть кількість гармонік";
			// 
			// label14
			// 
			this->label14->AutoSize = true;
			this->label14->Location = System::Drawing::Point(26, 136);
			this->label14->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label14->Name = L"label14";
			this->label14->Size = System::Drawing::Size(41, 15);
			this->label14->TabIndex = 17;
			this->label14->Text = L"Ng = ";
			// 
			// button2
			// 
			this->button2->ForeColor = System::Drawing::Color::SeaGreen;
			this->button2->Location = System::Drawing::Point(879, 103);
			this->button2->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(135, 58);
			this->button2->TabIndex = 8;
			this->button2->Text = L"Вийти";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &MyForm::button2_Click);
			// 
			// button1
			// 
			this->button1->ForeColor = System::Drawing::Color::SeaGreen;
			this->button1->Location = System::Drawing::Point(879, 28);
			this->button1->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(135, 69);
			this->button1->TabIndex = 7;
			this->button1->Text = L"Побудувати";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
			// 
			// numericUpDown3
			// 
			this->numericUpDown3->Location = System::Drawing::Point(758, 99);
			this->numericUpDown3->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->numericUpDown3->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
			this->numericUpDown3->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDown3->Name = L"numericUpDown3";
			this->numericUpDown3->Size = System::Drawing::Size(37, 21);
			this->numericUpDown3->TabIndex = 14;
			this->numericUpDown3->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			// 
			// numericUpDown2
			// 
			this->numericUpDown2->Location = System::Drawing::Point(758, 74);
			this->numericUpDown2->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->numericUpDown2->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
			this->numericUpDown2->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDown2->Name = L"numericUpDown2";
			this->numericUpDown2->Size = System::Drawing::Size(37, 21);
			this->numericUpDown2->TabIndex = 13;
			this->numericUpDown2->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			// 
			// numericUpDown1
			// 
			this->numericUpDown1->Location = System::Drawing::Point(758, 50);
			this->numericUpDown1->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->numericUpDown1->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
			this->numericUpDown1->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDown1->Name = L"numericUpDown1";
			this->numericUpDown1->Size = System::Drawing::Size(37, 21);
			this->numericUpDown1->TabIndex = 12;
			this->numericUpDown1->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			// 
			// textBox3
			// 
			this->textBox3->Location = System::Drawing::Point(75, 89);
			this->textBox3->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->textBox3->Name = L"textBox3";
			this->textBox3->Size = System::Drawing::Size(76, 21);
			this->textBox3->TabIndex = 3;
			// 
			// textBox2
			// 
			this->textBox2->Location = System::Drawing::Point(212, 41);
			this->textBox2->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->textBox2->Name = L"textBox2";
			this->textBox2->Size = System::Drawing::Size(76, 21);
			this->textBox2->TabIndex = 2;
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(75, 41);
			this->textBox1->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->textBox1->Name = L"textBox1";
			this->textBox1->Size = System::Drawing::Size(76, 21);
			this->textBox1->TabIndex = 1;
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Location = System::Drawing::Point(584, 99);
			this->label9->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(162, 15);
			this->label9->TabIndex = 8;
			this->label9->Text = L"Товщина ліній гратки =";
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Location = System::Drawing::Point(26, 41);
			this->label8->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(31, 15);
			this->label8->TabIndex = 7;
			this->label8->Text = L"al =";
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(163, 41);
			this->label7->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(35, 15);
			this->label7->TabIndex = 6;
			this->label7->Text = L"bl = ";
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(559, 79);
			this->label6->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(187, 15);
			this->label6->TabIndex = 5;
			this->label6->Text = L"Товщина осей координат =";
			this->label6->Click += gcnew System::EventHandler(this, &MyForm::label6_Click);
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(549, 52);
			this->label5->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(201, 15);
			this->label5->TabIndex = 4;
			this->label5->Text = L"Товщина лінії графіку ф-ції =";
			// 
			// label12
			// 
			this->label12->AutoSize = true;
			this->label12->ForeColor = System::Drawing::Color::SeaGreen;
			this->label12->Location = System::Drawing::Point(26, 69);
			this->label12->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label12->Name = L"label12";
			this->label12->Size = System::Drawing::Size(222, 15);
			this->label12->TabIndex = 3;
			this->label12->Text = L"Введіть кількість точок графіку";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(26, 89);
			this->label3->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(41, 15);
			this->label3->TabIndex = 2;
			this->label3->Text = L"Ne = ";
			// 
			// label11
			// 
			this->label11->AutoSize = true;
			this->label11->ForeColor = System::Drawing::Color::SeaGreen;
			this->label11->Location = System::Drawing::Point(549, 24);
			this->label11->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(249, 15);
			this->label11->TabIndex = 1;
			this->label11->Text = L"Оберіть, при потребі, інші значення";
			this->label11->Click += gcnew System::EventHandler(this, &MyForm::label11_Click);
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->ForeColor = System::Drawing::Color::SeaGreen;
			this->label10->Location = System::Drawing::Point(26, 24);
			this->label10->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(209, 15);
			this->label10->TabIndex = 0;
			this->label10->Text = L"Введіть межі зміни аргументу";
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1036, 613);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->pictureBox1);
			this->Margin = System::Windows::Forms::Padding(2, 2, 2, 2);
			this->Name = L"MyForm";
			this->Text = L"MyForm";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown4))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown3))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
		//double Kx, Ky, Zx, Zy; // коефiцiєнти масштабування
		//double Tp; // область визначення функції та її період : Tp = bl - al
		//double minYg, maxYg, maxx, maxy, minx, miny; // для обчислення коефіцієнтів масштабування
		//double krx, kry, xx, yy, Gx, Gy; // для виведення осей координат і їх підписів
		//int KrokX, KrokY, L;


		//double f(double x)
		//{
		//	switch (func_select->SelectedIndex)
		//	{
		//	case 0: return sin(x);
		//	case 1: return abs(((int)x % 2) - 1);
		//	case 2: return sin(x) + cos(2 * x);
		//	case 3: return abs(cos(x));
		//	default: MessageBox::Show("Виберіть функцію"); return 0;
		//	}
		//}

		//void TabF(double Xe[1000], double Ye[1000])
		//{
		//	double h;
		//	h = (bl - al) / Ne;
		//	Xe[0] = al;
		//	for (int i = 0; i <= Ne - 1; i++)
		//	{
		//		Ye[i] = f(Xe[i]);
		//		Xe[i + 1] = Xe[i] + h;
		//	}
		//}

		//void Furje(double Xe[1000], double Ye[1000], int Ne, double Yg[1000], double c[50], double TP)
		//{
		//	double a[50]; // масив a із коефіцієнтами ряду Фур’є
		//	double b[50]; // масив b із коефіцієнтами ряду Фур’є
		//	double w, KOM, S, G, D;
		//	if (Ng >= 50) {
		//		MessageBox::Show("Забагато гармонік. Функцію не розкладено у ряд Фур'є");
		//		return;
		//	}
		//	al = Convert::ToDouble(textBox1->Text);
		//	bl = Convert::ToDouble(textBox2->Text);
		//	TP = bl - al;
		//	w = 2 * Math::PI / TP;
		//	for (int k = 1; k <= Ng; k++) {
		//		KOM = k * w;
		//		G = 0;
		//		D = 0;
		//		switch (integral_select->SelectedIndex) {
		//		case 0:  // ?? метод
		//			for (int i = 1; i <= Ne - 1; i++) {
		//				S = KOM * Xe[i];
		//				G = G + Ye[i] * cos(S);
		//				D = D + Ye[i] * sin(S);
		//			}
		//			break;

		//		case 1:  // Метод прямокутників (центральний)
		//			for (int i = 1; i <= Ne - 1; i++) {
		//				S = KOM * Xe[i];
		//				G = G + Ye[i] * cos(S);
		//				D = D + Ye[i] * sin(S);
		//			}
		//			break;

		//		case 2:  // Метод трапецій
		//			for (int i = 1; i <= Ne - 1; i++) {
		//				S = KOM * Xe[i];
		//				G = G + 0.5 * (Ye[i] + Ye[i - 1]) * cos(S);
		//				D = D + 0.5 * (Ye[i] + Ye[i - 1]) * sin(S);
		//			}
		//			break;

		//		case 3:  // Метод парабол (Сімпсона)
		//			for (int i = 1; i <= Ne - 2; i += 2) {
		//				S = KOM * Xe[i];
		//				G = G + (Ye[i - 1] + 4 * Ye[i] + Ye[i + 1]) * cos(S) / 3.0;
		//				D = D + (Ye[i - 1] + 4 * Ye[i] + Ye[i + 1]) * sin(S) / 3.0;
		//			}
		//			break;
		//		default: MessageBox::Show("Виберіть метод обчислення інтегралу"); return;
		//		}
		//		a[k] = 2 * G / Ne;
		//		b[k] = 2 * D / Ne;
		//		c[k] = Math::Sqrt(a[k] * a[k] + b[k] * b[k]);
		//	}
		//	a[0] = 0;
		//	for (int i = 0; i <= Ne - 1; i++) {
		//		a[0] = a[0] + Ye[i];
		//	}
		//	a[0] = a[0] / Ne;
		//	for (int i = 0; i <= Ne - 1; i++) {
		//		S = 0;
		//		D = Xe[i] * w;
		//		for (int k = 1; k <= Ng; k++) {
		//			KOM = k * D;
		//			S = S + b[k] * sin(KOM) + a[k] * cos(KOM);
		//		}
		//		Yg[i] = a[0] + S;
		//	}
		//	return; // Завершення тіла функції Furje
		//}

		/*void Garm(int Ng, double c[50])
		{
			int i, KrokXG, x;
			double MaxC, KyC, w;
			Graphics^ g = pictureBox1->CreateGraphics();
			Pen^ pen1 = gcnew Pen(Color::Black, (float)(numericUpDown1->Value));
			Pen^ pen2 = gcnew Pen(Color::Blue, (float)(numericUpDown2->Value));
			Pen^ pen3 = gcnew Pen(Color::Silver, (float)(numericUpDown3->Value));
			Pen^ pen4 = gcnew Pen(Color::Green, (float)(numericUpDown4->Value));
			int pb_Height = pictureBox1->Height;
			int pb_Width = pictureBox1->Width;
			KrokXG = (pb_Width - 2 * L) / Ng;
			MaxC = c[1];
			for (i = 2; i <= Ng; i++)
				if (c[i] > MaxC) MaxC = c[i];
			KyC = (pb_Height / 2) / MaxC;
			g->DrawLine(pen2, L, L + 20, L + 10, L + 10);
			g->DrawLine(pen2, L + 20, L + 20, L + 10, L + 10);
			g->DrawLine(pen2, L + 10, pb_Height - 50, pb_Width - 20, pb_Height - 50);
			g->DrawLine(pen2, L + 10, pb_Height - 50, L + 10, L + 10);
			g->DrawLine(pen2, pb_Width - 40, pb_Height - 60, pb_Width - 20, pb_Height - 50);
			g->DrawString("C", gcnew Drawing::Font("Times", 14), Brushes::Black,
				(float)L - 15, (float)L + 5);
			g->DrawString("W", gcnew Drawing::Font("Times", 14), Brushes::Black, (float)pb_Width - 60.0f,
				(float)pb_Height - 50.0f);
			x = KrokXG + 20;
			w = 6.2831853 / (bl - al);
			for (i = 1; i <= Ng; i++)
			{
				g->DrawLine(pen1, (int)x + 3, pb_Height - 50, x + 3, pb_Height - 50 - (int)(KyC * c[i]));
				String^ s = String::Format("{0:F3}", KyC * c[i]);
				g->DrawString(s, gcnew Drawing::Font("Times", 12), Brushes::Black, (float)x,
					(float)pb_Height - (float)(KyC * c[i]) - 70.0f);
				g->DrawEllipse(pen2, (int)x, pb_Height - (int)(KyC * c[i]) - 55, 5, 5);
				g->DrawString(Convert::ToString(i), gcnew Drawing::Font("Times", 12), Brushes::Black,
					(float)x - 5.0f, (float)pb_Height - 50.0f);
				x = x + KrokXG;
			}
			x = KrokXG + 19;
			String^ s = String::Format("W={0:F3}", w);
			g->DrawString(s, gcnew Drawing::Font("Times", 12), Brushes::Black, (float)x - 20.0f,
				(float)pb_Height - 35.0f);
			return;
		}*/


	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
		Close();
	}
	private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
		
	/*	if (Ng < 50) {
			DialogResult = MessageBox::Show("Показати гармоніки?", "Гармоніки", MessageBoxButtons::YesNo, MessageBoxIcon::Question);
			if (DialogResult == System::Windows::Forms::DialogResult::Yes)
			{
				g->Clear(Color::White);
				Garm(Ng, c);
			}
		}*/
	}
	private: System::Void label11_Click(System::Object^ sender, System::EventArgs^ e) {
	}
	private: System::Void label6_Click(System::Object^ sender, System::EventArgs^ e) {
	}
	};
}