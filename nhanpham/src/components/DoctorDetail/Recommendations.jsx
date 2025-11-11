import { TrendingUp, CheckCircle } from 'lucide-react';

export const Recommendations = ({ recommendations }) => {
    return (
        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden flex flex-col">
            <div className="px-4 py-3 border-b border-white/10 bg-[#141414]">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-teal-500" />
                    Recommendations
                </h3>
            </div>
            <div className="p-3 max-h-[200px] overflow-y-auto custom-scrollbar">
                <ul className="space-y-1.5">
                    {recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start gap-2 text-xs text-gray-300 p-2 bg-[#0f0f0f] rounded border border-white/5">
                            <CheckCircle className="w-3.5 h-3.5 text-teal-500 mt-0.5 shrink-0" />
                            <span>{rec}</span>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};
