import { Brain } from 'lucide-react';

export const AIConfidence = ({ confidence }) => {
    return (
        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Brain className="w-4 h-4 text-teal-500" />
                    AI Confidence
                </h3>
                <span className="text-xl font-bold text-teal-500">{confidence}%</span>
            </div>
            <div className="w-full bg-white/10 rounded-full h-1.5 mb-1.5">
                <div
                    className="bg-teal-500 h-1.5 rounded-full transition-all"
                    style={{ width: `${confidence}%` }}
                ></div>
            </div>
            <p className="text-xs text-gray-500">Diagnostic Accuracy</p>
        </div>
    );
};