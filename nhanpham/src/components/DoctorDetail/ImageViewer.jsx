import { Calendar } from 'lucide-react';

export const ImageViewer = ({ image }) => {
    if (!image) {
        return (
            <div className="bg-[#1a1a1a] border border-white/10 rounded-xl flex items-center justify-center h-[calc(100vh-110px)]">
                <p className="text-gray-500">No image selected</p>
            </div>
        );
    }

    return (
        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden flex flex-col h-[calc(100vh-110px)]">
            {/* Header */}
            <div className="px-4 py-3 border-b border-white/10 bg-[#141414]">
                <div className="flex items-center justify-between">
                    <h3 className="text-base font-semibold text-white">{image.type}</h3>
                    <div className="flex items-center gap-3 mt-1">
                        <span className="text-xs text-gray-400 flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            {new Date().toLocaleDateString()}
                        </span>
                        <span className="text-xs px-2 py-0.5 bg-teal-500/20 border border-teal-500/30 rounded text-teal-400">
                            {image.modality}
                        </span>
                        <span className="text-xs text-gray-500">Code: {image.imageCode}</span>
                    </div>
                </div>
            </div>

            {/* Image Container */}
            <div className="flex-1 flex items-center justify-center bg-black/30 p-4 overflow-hidden">
                <img
                    src={image.url}
                    alt={image.type}
                    className="max-w-full max-h-full object-contain"
                />
            </div>
        </div>
    );
};