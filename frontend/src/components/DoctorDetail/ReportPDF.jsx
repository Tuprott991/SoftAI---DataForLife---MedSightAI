import { FileText, Download, Printer } from 'lucide-react';

export const ReportPDF = ({ reportData, patient }) => {
    if (!reportData || !patient) return null;

    const handlePrint = () => {
        window.print();
    };

    const handleDownload = () => {
        // TODO: Implement PDF download
        alert('Chức năng tải xuống sẽ được triển khai sau');
    };

    return (
        <div className="bg-white text-black">
            {/* Action Buttons - Hidden when printing */}
            <div className="flex gap-2 mb-4 print:hidden">
                <button
                    onClick={handlePrint}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                    <Printer className="w-4 h-4" />
                    <span>In báo cáo</span>
                </button>
                <button
                    onClick={handleDownload}
                    className="flex items-center gap-2 px-4 py-2 bg-teal-500 text-white rounded-lg hover:bg-teal-600 transition-colors"
                >
                    <Download className="w-4 h-4" />
                    <span>Tải xuống</span>
                </button>
            </div>

            {/* PDF Content */}
            <div className="bg-white p-8 max-w-[210mm] mx-auto shadow-lg">
                {/* Header - Hospital Info */}
                <div className="relative mb-6 pb-4">
                    <div className="border-2 border-gray-400 rounded-lg p-8 relative">
                        <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-white px-3">
                            <span className="text-xs font-semibold text-gray-600 uppercase">Thông tin bệnh viện</span>
                        </div>
                        <div className="text-center mt-2">
                            <p className="text-sm text-gray-500 italic">Đây là chỗ xem thông tin bệnh viện</p>
                        </div>
                    </div>
                </div>

                {/* Title */}
                <div className="text-center mb-6">
                    <h1 className="text-2xl font-bold text-gray-800 mb-2">BÁO CÁO CHẨN ĐOÁN HÌNH ẢNH</h1>
                    <div className="text-sm text-gray-600">KHOA CHẨN ĐOÁN HÌNH ẢNH</div>
                </div>

                {/* Patient Information */}
                <div className="mb-6">
                    <h2 className="text-lg font-semibold text-gray-800 mb-3 pb-2 border-b border-gray-300">
                        I. THÔNG TIN BỆNH NHÂN
                    </h2>
                    <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
                        <div className="flex">
                            <span className="font-semibold w-32">Họ và tên:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.ho_ten}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">Tuổi:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.tuoi}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">Giới tính:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.gioi_tinh}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">Nhóm máu:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.nhom_mau}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">Ngày chụp:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.ngay_chup}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">Ngày đọc phim:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.ngay_doc_phim}</span>
                        </div>
                        <div className="flex col-span-2">
                            <span className="font-semibold w-32">Chẩn đoán LS:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.chan_doan_lam_sang}</span>
                        </div>
                    </div>
                </div>

                {/* Report Details */}
                <div className="mb-6">
                    <h2 className="text-lg font-semibold text-gray-800 mb-3 pb-2 border-b border-gray-300">
                        II. BÁO CÁO X-QUANG
                    </h2>
                    
                    <div className="space-y-3 text-sm">
                        <div>
                            <span className="font-semibold">MeSH Tags: </span>
                            <span className="text-gray-700">{reportData.bao_cao_x_quang.MeSH}</span>
                        </div>

                        <div>
                            <span className="font-semibold">Loại ảnh: </span>
                            <span className="text-gray-700">{reportData.bao_cao_x_quang.loai_anh}</span>
                        </div>

                        <div>
                            <span className="font-semibold">Chỉ định: </span>
                            <span className="text-gray-700">{reportData.bao_cao_x_quang.chi_dinh}</span>
                        </div>

                        <div>
                            <span className="font-semibold">So sánh: </span>
                            <span className="text-gray-700">{reportData.bao_cao_x_quang.so_sanh}</span>
                        </div>
                    </div>
                </div>

                {/* Description */}
                <div className="mb-6">
                    <h3 className="text-base font-semibold text-gray-800 mb-2">Mô tả:</h3>
                    <p className="text-sm text-gray-700 leading-relaxed text-justify">
                        {reportData.bao_cao_x_quang.mo_ta}
                    </p>
                </div>

                {/* Conclusion */}
                <div className="mb-8">
                    <h3 className="text-base font-semibold text-gray-800 mb-2">Kết luận:</h3>
                    <p className="text-sm text-gray-700 leading-relaxed text-justify">
                        {reportData.bao_cao_x_quang.ket_luan}
                    </p>
                </div>

                {/* Signature Section */}
                <div className="mt-12 pt-6">
                    <div className="grid grid-cols-2 gap-4">
                        {/* Left signature box */}
                        <div className="relative">
                            <div className="border-2 border-gray-400 rounded-lg p-8 relative h-32 flex items-center justify-center">
                                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-white px-3">
                                    <span className="text-xs font-semibold text-gray-600 uppercase">Người nhận báo cáo</span>
                                </div>
                                <p className="text-sm text-gray-500 italic text-center">Đây là chỗ chữ ký của người nhận</p>
                            </div>
                        </div>

                        {/* Right signature box */}
                        <div className="relative">
                            <div className="border-2 border-gray-400 rounded-lg p-8 relative h-32 flex items-center justify-center">
                                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-white px-3">
                                    <span className="text-xs font-semibold text-gray-600 uppercase">Bác sĩ chẩn đoán</span>
                                </div>
                                <p className="text-sm text-gray-500 italic text-center">Đây là chỗ chữ ký của bác sĩ</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="mt-6 pt-4 border-t border-gray-200 text-center text-xs text-gray-500">
                    <p>Báo cáo này được tạo tự động bởi hệ thống MedSightAI</p>
                    <p>Vui lòng liên hệ bác sĩ điều trị để được tư vấn chi tiết</p>
                </div>
            </div>

            {/* Print styles */}
            <style jsx>{`
                @media print {
                    @page {
                        size: A4;
                        margin: 15mm;
                    }
                    
                    body {
                        print-color-adjust: exact;
                        -webkit-print-color-adjust: exact;
                    }
                    
                    .print\\:hidden {
                        display: none !important;
                    }
                }
            `}</style>
        </div>
    );
};
