#include <iostream>
#include <string>

static std::string data_path = "merged_data.root";

// Example mc_path and effq_path definitions (you may want to uncomment/set as needed)
// static std::string mc_path = "merged_hits.root";
// static std::string effq_path = "merged_effq.root";
// std::string label = "";
/* static std::string mc_path = "merged_shield_delay28_hits.root"; */
/* static std::string effq_path = "merged_shield_delay28_effq.root"; */
/* const std::string label = "shield_delay28"; */
/* static std::string mc_path = "merged_noshield_thres5k_hits.root"; */
/* static std::string effq_path = "merged_noshield_thres5k_effq.root"; */
/* const std::string label = "noshield_thres5k"; */
static std::string mc_path = "merged_shield_hits.root";
static std::string effq_path = "merged_shield_effq.root";
const std::string label = "shield";

TH1F* draw_from_tree(std::string filename, std::string treename, std::string var, std::string sel, int n=45, float nmin=0, float nmax=45) {
  std::cout << filename << std::endl;
  std::cout << treename << std::endl;
  std::cout << var << std::endl;
  std::cout << sel << std::endl;
    TFile *file = TFile::Open(filename.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file!" << std::endl;
        return nullptr;
    }
    TTree *tree = (TTree*)file->Get(treename.c_str());
    if (!tree) {
        std::cerr << "Error: Tree not found!" << std::endl;
        return nullptr;
    }
    TH1F *h1d = new TH1F("h1d", "", n, nmin, nmax);
    tree->Draw(::Form("%s>>h1d", var.c_str()), sel.c_str());
    h1d->SetDirectory(0);
    file->Close();
    return h1d;
}

double sum_total_length(std::string filename, std::string treename){
    auto df = ROOT::RDataFrame(treename, filename);
    return df.Sum("distance").GetValue();
}

void draw_dx(const std::string label){
    auto hdx_2x2 = draw_from_tree(data_path, "selected_data/hits", "dx", "", 100, -2,3);
    hdx_2x2->SetName("hdx_2x2");
    auto hdx_tred = draw_from_tree(mc_path, "selected_data/hits", "dx", "", 100,-2,3);
    hdx_tred->SetName("hdx_tred");
    TCanvas* c1 = new TCanvas("cdx", "dx", 800, 600);
    hdx_2x2->SetTitle(";dx;normalized counts");
    hdx_2x2->Scale(1./hdx_2x2->GetEntries());
    hdx_tred->Scale(1./hdx_tred->GetEntries());
    hdx_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(hdx_2x2->GetMaximum(), hdx_tred->GetMaximum()));
    hdx_2x2->Draw();
    hdx_2x2->SetLineColor(kRed);
    hdx_tred->Draw("SAME");
    hdx_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(hdx_2x2, "2x2");
    leg->AddEntry(hdx_tred, "tred");
    leg->Draw();
    if (label != "") {
        c1->Print(::Form("comp_dx_%s.png", label.c_str()));
    } else {
        c1->Print("comp_dx.png");
    }
}

void draw_totQ(const bool uselength, const bool useqeff, const std::string label){

    auto htotQ_2x2 = draw_from_tree(data_path, "selected_data/hits", "totQ", "tindex == 0");
    htotQ_2x2->SetName("htotQ_2x2");
    auto htotQ_tred = draw_from_tree(mc_path, "selected_data/hits", "totQ", "tindex == 0");
    htotQ_tred->SetName("htotQ_tred");
    auto htotQ_effq = draw_from_tree(effq_path, "selected_data/hits", "totQ.", "tindex == 0 && totQ > 5");
    auto htotQ_effq2 = draw_from_tree(effq_path, "selected_data/hits", "totQ.", "tindex == 0 && totQ > 3");
    auto htotQ_effq3 = draw_from_tree(effq_path, "selected_data/hits", "totQ.", "tindex == 0 && totQ > 7");
    htotQ_effq->SetName("htotQ_effq");
    htotQ_effq2->SetName("htotQ_effq2");
    htotQ_effq3->SetName("htotQ_effq3");
    TCanvas* c1 = new TCanvas("ctotQ", "totQ", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        auto d_effq = sum_total_length(effq_path, "selected_data/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_tred->Scale(1./d_tred);
        if (useqeff && uselength) {
            htotQ_effq->Scale(1./d_effq);
            htotQ_effq2->Scale(1./d_effq);
            htotQ_effq3->Scale(1./d_effq);
        }
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);

    if (useqeff && uselength) {
        htotQ_effq->Draw("HIST SAME");
        htotQ_effq->SetLineColor(kGreen-3);
        htotQ_effq->SetLineStyle(kDashed);

        htotQ_effq2->Draw("HIST SAME");
        htotQ_effq2->SetLineColor(kGreen+3);
        htotQ_effq2->SetLineStyle(3);

        htotQ_effq3->Draw("HIST SAME");
        htotQ_effq3->SetLineColor(kGreen-5);
        htotQ_effq3->SetLineStyle(4);
    }

    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    if (useqeff && uselength) {
        leg->AddEntry(htotQ_effq, "effq; totQ>5ke-");
        leg->AddEntry(htotQ_effq2, "effq; totQ>3ke-");
        leg->AddEntry(htotQ_effq3, "effq; totQ>7ke-");
    }
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
    }
    if (label != "") {
        if (uselength) {
            c1->Print(::Form("comp_totQ_norm_by_l_%s.png", label.c_str()));
        } else {
            c1->Print(::Form("comp_totQ_%s.png", label.c_str()));
        }
    } else {
        if (uselength) {
            c1->Print("comp_totQ_norm_by_l.png");
        } else {
            c1->Print("comp_totQ.png");
        }
    }
}

void draw_totQ_smeared(const bool uselength, const std::string label){
    auto htotQ_2x2 = draw_from_tree(data_path, "selected_data/hits", "totQ", "tindex == 0");
    htotQ_2x2->SetName("htotQ_2x2");
    auto htotQ_tred = draw_from_tree(mc_path, "selected_data/hits", "totQ", "tindex == 0");
    htotQ_tred->SetName("htotQ_tred");
    TCanvas* c1 = new TCanvas("ctotQ_smeared", "totQ_smeared", 800, 600);
    ROOT::RDataFrame df("selected_data/hits", mc_path);
    auto dfnew = df.Define("totQ_smear1k", [](const double totQ)->double { return totQ + gRandom->Uniform(-0.5, 0.5); }, {"totQ"}).
    Define("totQ_smear2k", [](const double totQ)->double{ return totQ + gRandom->Uniform(-0.5, 0.5) * 2; }, {"totQ"}).
    Define("totQ_smear4k", [](const double totQ)->double{ return totQ + gRandom->Uniform(-0.5, 0.5) * 4; }, {"totQ"});
    auto htotQ_smear1k = dfnew.Filter("tindex == 0").Histo1D({"htotQ_smear1k", "totQ + Uniform[-0.5, 0.5]", 50, 0, 50}, "totQ_smear1k");
    auto htotQ_smear2k = dfnew.Filter("tindex == 0").Histo1D({"htotQ_smear2k", "totQ + Uniform[-1, 1]", 50, 0, 50}, "totQ_smear2k");
    auto htotQ_smear4k = dfnew.Filter("tindex == 0").Histo1D({"htotQ_smear2k", "totQ + Uniform[-2, 2]", 50, 0, 50}, "totQ_smear4k");
    htotQ_2x2->SetTitle("total Q per pixel;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_tred->Scale(1./d_tred);
        htotQ_smear1k->Scale(1./d_tred);
        htotQ_smear2k->Scale(1./d_tred);
        htotQ_smear4k->Scale(1./d_tred);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_tred->Scale(1./htotQ_tred->GetEntries());
        htotQ_smear1k->Scale(1./htotQ_smear1k->GetEntries());
        htotQ_smear2k->Scale(1./htotQ_smear2k->GetEntries());
        htotQ_smear4k->Scale(1./htotQ_smear4k->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    htotQ_smear1k->SetLineColor(kGreen-3);
    htotQ_smear2k->SetLineColor(kGreen+3);
    htotQ_smear4k->SetLineColor(kGreen+5);
    htotQ_smear1k->SetLineStyle(2);
    htotQ_smear2k->SetLineStyle(3);
    htotQ_smear4k->SetLineStyle(4);
    htotQ_smear1k->Draw("SAME HIST");
    htotQ_smear2k->Draw("SAME HIST");
    htotQ_smear4k->Draw("SAME HIST");

    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->AddEntry(htotQ_smear1k.GetPtr(), "tred + Uniform[-0.5,0.5]");
    leg->AddEntry(htotQ_smear2k.GetPtr(), "tred + Uniform[-1,1]");
    leg->AddEntry(htotQ_smear4k.GetPtr(), "tred + Uniform[-2,2]");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
    }
    if (label != "") {
        if (uselength) {
            c1->Print(::Form("comp_totQ_smear_norm_by_l_%s.png", label.c_str()));
        } else {
            c1->Print(::Form("comp_totQ_smear_%s.png", label.c_str()));
        }
    } else {
        if (uselength) {
            c1->Print("comp_totQ_smear_norm_by_l.png");
        } else {
            c1->Print("comp_totQ_smear.png");
        }
    }
}

void draw_totN(const bool uselength, const std::string label){
    auto htotN_2x2 = draw_from_tree(data_path, "selected_data/hits", "totN", "tindex == 0", 5, -0.5, 4.5);
    htotN_2x2->SetName("htotN_2x2");
    auto htotN_tred = draw_from_tree(mc_path, "selected_data/hits", "totN", "tindex == 0", 5, -0.5, 4.5);
    htotN_tred->SetName("htotN_tred");
    TCanvas* c2 = new TCanvas("ctotN", "totN", 800, 600);
    htotN_2x2->SetTitle("total N per pixel;totN;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        htotN_2x2->Scale(1./d_2x2);
        htotN_tred->Scale(1./d_tred);
    } else {
        htotN_2x2->Scale(1./htotN_2x2->GetEntries());
        htotN_tred->Scale(1./htotN_tred->GetEntries());
    }
    htotN_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotN_2x2->GetMaximum(), htotN_tred->GetMaximum()));
    htotN_2x2->Draw();
    htotN_2x2->SetLineColor(kRed);
    htotN_tred->Draw("SAME");
    htotN_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotN_2x2, "2x2");
    leg->AddEntry(htotN_tred, "tred");
    leg->Draw();
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.5, 0.55, ::Form("Integral (2x2): %.2f * %.0fcm", htotN_2x2->Integral(), d_2x2));
        tex->DrawLatexNDC(0.5, 0.5, ::Form("Integral (tred): %.2f * %.0fcm", htotN_tred->Integral(), d_tred));
    }
    if (label != "") {
        if (uselength) {
            c2->Print(::Form("comp_totN_norm_by_l_%s.png", label.c_str()));
        } else {
            c2->Print(::Form("comp_totN_%s.png", label.c_str()));
        }
    } else {
        if (uselength) {
            c2->Print("comp_totN_norm_by_l.png");
        } else {
            c2->Print("comp_totN.png");
        }
    }
}

void draw_totN_totQ30(const bool uselength, const std::string label){
    auto htotN_2x2 = draw_from_tree(data_path, "selected_data/hits", "totN", "tindex == 0 && totQ>30", 5, -0.5, 4.5);
    htotN_2x2->SetName("htotNtotQ30_2x2");
    auto htotN_tred = draw_from_tree(mc_path, "selected_data/hits", "totN", "tindex == 0 && totQ>30", 5, -0.5, 4.5);
    htotN_tred->SetName("htotNtotQ30_tred");
    TCanvas* c2 = new TCanvas("ctotN_totQ30", "totN_totQ30", 800, 600);
    htotN_2x2->SetTitle("total N per pixel,totQ>30;totN;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        htotN_2x2->Scale(1./d_2x2);
        htotN_tred->Scale(1./d_tred);
    } else {
        htotN_2x2->Scale(1./htotN_2x2->GetEntries());
        htotN_tred->Scale(1./htotN_tred->GetEntries());
    }
    htotN_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotN_2x2->GetMaximum(), htotN_tred->GetMaximum()));
    htotN_2x2->Draw();
    htotN_2x2->SetLineColor(kRed);
    htotN_tred->Draw("SAME");
    htotN_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotN_2x2, "2x2");
    leg->AddEntry(htotN_tred, "tred");
    leg->Draw();
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.5, 0.55, ::Form("Integral (2x2): %.2f * %.0fcm", htotN_2x2->Integral(), d_2x2));
        tex->DrawLatexNDC(0.5, 0.5, ::Form("Integral (tred): %.2f * %.0fcm", htotN_tred->Integral(), d_tred));
    }
    if (label != "") {
        if (uselength) {
            c2->Print(::Form("comp_totNtotQ30_norm_by_l_%s.png", label.c_str()));
        } else {
            c2->Print(::Form("comp_totNtotQ30_%s.png", label.c_str()));
        }
    } else {
        if (uselength) {
            c2->Print("comp_totNtotQ30_norm_by_l.png");
        } else {
            c2->Print("comp_totNtotQ30.png");
        }
    }
}

void draw_totQ_totN1(const bool uselength, const std::string label){
    auto htotQ_2x2 = draw_from_tree(data_path, "selected_data/hits", "totQ", "tindex == 0 && totN == 1");
    htotQ_2x2->SetName("htotQ_2x2_totN1");
    auto htotQ_tred = draw_from_tree(mc_path, "selected_data/hits", "totQ", "tindex == 0 && totN==1");
    htotQ_tred->SetName("htotQ_tred_totN1");
    TCanvas* c1 = new TCanvas("ctotQtotN1", "totQtotN1", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==1;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_tred->Scale(1./d_tred);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.2, 0.75, ::Form("Integral (2x2): %.2f", htotQ_2x2->Integral()));
        tex->DrawLatexNDC(0.2, 0.68, ::Form("Integral (tred): %.2f", htotQ_tred->Integral()));
    }
    if (label != "") {
        if (uselength) {
            c1->Print(::Form("comp_totQ_totN1_norm_by_l_%s.png", label.c_str()));
        } else {
            c1->Print(::Form("comp_totQ_totN1_%s.png", label.c_str()));
        }
    } else {
        if (uselength) {
            c1->Print("comp_totQ_totN1_norm_by_l.png");
        } else {
            c1->Print("comp_totQ_totN1.png");
        }
    }
}

void draw_totQ_totN2(const bool uselength, const std::string label){
    auto htotQ_2x2 = draw_from_tree(data_path, "selected_data/hits", "totQ", "tindex == 0 && totN == 2");
    htotQ_2x2->SetName("htotQ_2x2_totN2");
    auto htotQ_tred = draw_from_tree(mc_path, "selected_data/hits", "totQ", "tindex == 0 && totN==2");
    htotQ_tred->SetName("htotQ_tred_totN2");
    TCanvas* c1 = new TCanvas("ctotQtotN2", "totQtotN2", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==2;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_tred->Scale(1./d_tred);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.2, 0.75, ::Form("Integral (2x2): %.2f", htotQ_2x2->Integral()));
        tex->DrawLatexNDC(0.2, 0.68, ::Form("Integral (tred): %.2f", htotQ_tred->Integral()));
    }
    if (label != "") {
        if (uselength) {
            c1->Print(::Form("comp_totQ_totN2_norm_by_l_%s.png", label.c_str()));
        } else {
            c1->Print(::Form("comp_totQ_totN2_%s.png", label.c_str()));
        }
    } else {
        if (uselength) {
            c1->Print("comp_totQ_totN2_norm_by_l.png");
        } else {
            c1->Print("comp_totQ_totN2.png");
        }
    }
}

void draw_totQ_totN3(const bool uselength, const std::string label){
    auto htotQ_2x2 = draw_from_tree(data_path, "selected_data/hits", "totQ", "tindex == 0 && totN == 3");
    htotQ_2x2->SetName("htotQ_2x2_totN3");
    auto htotQ_tred = draw_from_tree(mc_path, "selected_data/hits", "totQ", "tindex == 0 && totN==3");
    htotQ_tred->SetName("htotQ_tred_totN3");
    TCanvas* c1 = new TCanvas("ctotQtotN3", "totQtotN3", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==3;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_tred->Scale(1./d_tred);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.2, 0.75, ::Form("Integral (2x2): %.2f", htotQ_2x2->Integral()));
        tex->DrawLatexNDC(0.2, 0.68, ::Form("Integral (tred): %.2f", htotQ_tred->Integral()));
    }
    if (label != "") {
        if (uselength) {
            c1->Print(::Form("comp_totQ_totN3_norm_by_l_%s.png", label.c_str()));
        } else {
            c1->Print(::Form("comp_totQ_totN3_%s.png", label.c_str()));
        }
    } else {
        if (uselength) {
            c1->Print("comp_totQ_totN3_norm_by_l.png");
        } else {
            c1->Print("comp_totQ_totN3.png");
        }
    }
}

void draw_thres(const bool uselength, const std::string label){
    auto hthres_2x2 = draw_from_tree(data_path, "selected_data/hits", "thres", "tindex == 0");
    hthres_2x2->SetName("hthres_2x2");
    auto hthres_tred = draw_from_tree(mc_path, "selected_data/hits", "thres", "tindex == 0");
    hthres_tred->SetName("hthres_tred");
    TCanvas* c1 = new TCanvas("cthres", "thres", 800, 600);
    hthres_2x2->SetTitle("frequency of thresholds at each triggered channel;thres;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length(data_path, "selected_data/distances");
        auto d_tred = sum_total_length(mc_path, "selected_data/distances");
        hthres_2x2->Scale(1./d_2x2);
        hthres_tred->Scale(1./d_tred);
    } else {
        hthres_2x2->Scale(1./hthres_2x2->GetEntries());
        hthres_tred->Scale(1./hthres_tred->GetEntries());
    }
    hthres_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(hthres_2x2->GetMaximum(), hthres_tred->GetMaximum()));
    hthres_2x2->Draw();
    hthres_2x2->SetLineColor(kRed);
    hthres_tred->Draw("SAME");
    hthres_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(hthres_2x2, "2x2");
    leg->AddEntry(hthres_tred, "tred");
    leg->Draw();
    if (label != "") {
        if (uselength) {
            c1->Print(::Form("comp_thres_norm_by_l_%s.png", label.c_str()));
        } else {
            c1->Print(::Form("comp_thres_%s.png", label.c_str()));
        }
    } else {
        if (uselength) {
            c1->Print("comp_thres_norm_by_l.png");
        } else {
            c1->Print("comp_thres.png");
        }
    }
}

void draw_totQ_totN(bool uselength=false){
    TH1::SetDefaultSumw2();
    gStyle->SetOptStat(0);
    draw_totQ(uselength, true, label);
    draw_totN(uselength, label);
    draw_totN_totQ30(uselength, label);
    draw_totQ_totN1(uselength, label);
    draw_totQ_totN2(uselength, label);
    draw_totQ_totN3(uselength, label);
    draw_thres(uselength, label);
    draw_dx(label);
    draw_totQ_smeared(uselength, label);
}
