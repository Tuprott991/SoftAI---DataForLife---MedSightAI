import { KeyFindings } from '../KeyFindings';
import { Measurements } from '../Measurements';

export const AnalysisTab = ({ findings, metrics }) => {
    return (
        <div className="space-y-4">
            <KeyFindings findings={findings} />
            <Measurements metrics={metrics} />
        </div>
    );
};
