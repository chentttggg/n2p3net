"""One owned CPU programme actor with explicit local or cloud resource scope.

Both scopes retain the existing CloudExecutor/HostMemoryAdmission owner and
same-process scientific registries. Only cloud capacity requires the CUDA floor;
the actual population programme always consumes a CPU phase and zero GPU work.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
from research.execution import atomic_write_json
from research.source_execution import (
    CLOUD_RESOURCE_CONFIG,
    HostMemoryAdmission,
    available_host_memory_bytes,
    resolve_source_resources,
    with_cloud_resource_floor,
)

RESOURCE_SCHEMAS = {'cloud': 'public_population_source_resources/3',
                    'local_development': 'public_population_source_local_resources/1'}


def programme_requirements(resources):
    scope = resources.get('resource_scope')
    if (scope not in RESOURCE_SCHEMAS or resources.get('schema') != RESOURCE_SCHEMAS[scope]
            or resources.get('device') != 'cpu'):
        raise ValueError('source fitting requires an explicit owned local or cloud CPU declaration')
    names = ('source_workers', 'fold_workers', 'native_threads_per_worker', 'cpu_threads_total',
             'source_worker_reservation_bytes', 'fold_worker_reservation_bytes', 'shared_programme_reservation_bytes')
    if any(type(resources.get(name)) is not int or resources[name] < 1 for name in names):
        raise ValueError('programme CPU widths and complete RAM reservations must be positive integers')
    concurrency = max(resources['source_workers'], resources['fold_workers'])
    native = resources['native_threads_per_worker']
    if concurrency > 13 or concurrency * native + 1 > resources['cpu_threads_total']:
        raise ValueError('the whole source programme must fit its pool and central controller CPU budget')
    if scope == 'cloud':
        headroom = resources.get('memory_headroom_fraction')
        if isinstance(headroom, bool) or not isinstance(headroom, (float, int)) or not .05 <= headroom < 1:
            raise ValueError('the cloud programme requires at least five percent effective host headroom')
    elif (resources.get('whole_machine_utilization_ceiling') != .8
            or type(resources.get('programme_memory_limit_bytes')) is not int
            or resources['programme_memory_limit_bytes'] < 1 or resources.get('cloud_resource_floor') is not None
            or not isinstance(resources.get('resource_observation'), dict)
            or not resources['resource_observation'].get('source')):
        raise ValueError('local research needs its explicit eighty-percent policy, observed budget and sampling source')
    memory = max(resources['source_workers'] * resources['source_worker_reservation_bytes'],
                 resources['fold_workers'] * resources['fold_worker_reservation_bytes'])
    memory += resources['shared_programme_reservation_bytes']
    return {'resource_scope': scope, 'concurrency': concurrency, 'native_threads': native,
            'programme_cpu_threads': concurrency * native, 'programme_memory_bytes': memory,
            'ready_gpu_tasks': 0, 'programme_gpu_claim': 0}


def local_memory_budget(resources, *, total_bytes, available_bytes, effective_available_bytes=None):
    """Pure scalar admission against the current whole-machine RAM headroom."""
    if (type(total_bytes) is not int or type(available_bytes) is not int
            or not 0 < available_bytes <= total_bytes):
        raise ValueError('actual host total and available RAM observations are required')
    used = total_bytes - available_bytes
    remaining = max(0, int(.8 * total_bytes) - 1 - used)
    effective = available_bytes if effective_available_bytes is None else effective_available_bytes
    if type(effective) is not int or not 0 < effective <= available_bytes:
        raise ValueError('effective host availability must fit the actual physical observation')
    budget = min(resources['programme_memory_limit_bytes'], remaining, effective)
    if budget <= 0:
        raise ValueError('the host has no available research RAM below the eighty-percent ceiling')
    return {'host_total_bytes': total_bytes, 'host_nonavailable_bytes': used,
            'host_available_bytes': available_bytes, 'whole_machine_utilization_ceiling': .8,
            'effective_host_available_bytes': effective,
            'actual_host_remaining_below_ceiling_bytes': remaining,
            'declared_programme_memory_limit_bytes': resources['programme_memory_limit_bytes'],
            'effective_programme_memory_limit_bytes': budget,
            'effective_host_memory_fraction': budget / available_bytes,
            'declared_resource_observation': resources['resource_observation']}


def resolve_programme_owner(resources, *, cpu_count=None, free_device_bytes=None, free_host_bytes=None,
                            host_total_bytes=None, effective_host_bytes=None):
    """Resolve declared topology and actual resource capacity; never downgrade scope."""
    requested = programme_requirements(resources)
    maximum = max(resources['source_worker_reservation_bytes'], resources['fold_worker_reservation_bytes'])
    scope = requested['resource_scope']
    if scope == 'cloud':
        central = json.loads(CLOUD_RESOURCE_CONFIG.read_text(encoding='utf8'))
        plan = {**central, 'resource_scope': 'cloud', 'cpu_budget': resources['cpu_threads_total'],
                'source_workers': requested['concurrency'], 'target_workers': central['cloud_resource_floor']['gpu_workers'],
                'source_worker_memory_gib': maximum / 2**30, 'shared_staging_memory_gib': 0,
                'host_memory_fraction': 1 - resources['memory_headroom_fraction']}
        if resources.get('cloud_resource_floor') is not None:
            plan['cloud_resource_floor'] = resources['cloud_resource_floor']
        capacity = resolve_source_resources(with_cloud_resource_floor(plan, 'cuda'), 13, 'cuda',
            cpu_count=cpu_count, free_device_bytes=free_device_bytes, free_host_bytes=free_host_bytes)
        capacity['programme_capacity_device'] = 'cuda'
    else:
        observed_runtime = host_total_bytes is None or free_host_bytes is None
        if observed_runtime:
            import psutil

            observed = psutil.virtual_memory()
            host_total_bytes = int(observed.total) if host_total_bytes is None else host_total_bytes
            free_host_bytes = int(observed.available) if free_host_bytes is None else free_host_bytes
        effective = (min(free_host_bytes, available_host_memory_bytes()) if observed_runtime
                     else free_host_bytes) if effective_host_bytes is None else effective_host_bytes
        memory = local_memory_budget(resources, total_bytes=host_total_bytes, available_bytes=free_host_bytes,
                                     effective_available_bytes=effective)
        aggregate_gib = requested['programme_memory_bytes'] / 2**30
        plan = {'resource_scope': 'local_development', 'cpu_budget': resources['cpu_threads_total'],
                'source_workers': 1, 'target_workers': 1,
                'source_worker_memory_gib': aggregate_gib, 'target_worker_memory_gib': aggregate_gib,
                'shared_staging_memory_gib': 0, 'host_memory_fraction': memory['effective_host_memory_fraction']}
        # There is one process and one aggregate lease. Its internal thread
        # pools consume the phase's explicit CPU/memory claim below.
        capacity = resolve_source_resources(plan, 1, 'cpu', cpu_count=cpu_count, free_host_bytes=free_host_bytes)
        # The integer declaration is the authoritative ledger budget; converting
        # through a fraction may round the resolver's result by one byte.
        capacity['host_memory_budget_bytes'] = memory['effective_programme_memory_limit_bytes']
        capacity['local_host_memory_admission'] = {**memory,
            'actual_observation_source': 'psutil.virtual_memory_at_admission' if observed_runtime else 'explicit_scalar_metadata_control'}
        capacity['programme_capacity_device'] = 'cpu'
        capacity['gpu_slot_semantics'] = 'CPU admission slots; no GPU backing or GPU programme claim'
    capacity['programme_resource_scope'] = scope
    if (capacity['cpu_budget'] != resources['cpu_threads_total']
            or scope == 'cloud' and capacity['source_workers'] < requested['concurrency']
            or requested['programme_cpu_threads'] + capacity['coordinator_cpu_threads'] > capacity['cpu_budget']
            or requested['programme_memory_bytes'] > capacity['host_memory_budget_bytes'] - capacity['shared_staging_memory_bytes']):
        raise ValueError('the actual shared owner cannot admit the complete source programme phase')
    phase = PhaseResources('source', requested['programme_cpu_threads'], requested['programme_memory_bytes'],
                           requested['native_threads'], 'cpu')
    return capacity, phase


def require_programme_phase(resources, capacity, managed):
    requested = programme_requirements(resources)
    phase = managed.task.resources
    if (capacity['programme_resource_scope'] != requested['resource_scope']
            or managed.admission.cpu_budget != capacity['cpu_budget']
            or managed.admission.gpu_worker_limit != capacity['gpu_slots']
            or resources['cpu_threads_total'] != capacity['cpu_budget']
            or phase.phase != 'source' or phase.device != 'cpu'
            or phase.native_threads != requested['native_threads']
            or phase.cpu_threads < requested['programme_cpu_threads']
            or phase.memory_bytes < requested['programme_memory_bytes']):
        raise ValueError('the actual programme phase does not cover its explicit scope and internal source/fold pools')
    return {'resource_scope': requested['resource_scope'], 'project_capacity_device': capacity['programme_capacity_device'],
            'actual_programme_device': 'cpu', 'actual_programme_phase': asdict(phase), 'ready_gpu_tasks': 0,
            'programme_gpu_claim': 0, 'internal_source_workers': resources['source_workers'],
            'internal_fold_workers': resources['fold_workers'],
            'member_leases': 'covered by the outer aggregate phase; no nested double reservation'}


def execute_population_programme(arguments):
    """The scientific entry runs in one original managed local/cloud process."""
    from experiments.run_public_population_source import main

    return main(arguments['argv'], cloud_owner_resources=arguments['capacity'])


def run_population_source_cloud(*, argv, resources, output, identity):
    """Use the existing family actor API with the explicitly declared host scope."""
    scope = programme_requirements(resources)['resource_scope']
    output = Path(output).resolve()
    controller = output.with_name(output.name + ('.cloud' if scope == 'cloud' else '.local'))
    if output.exists() or controller.exists():
        raise FileExistsError('the source programme requires new scientific and controller output directories')
    capacity, phase = resolve_programme_owner(resources)
    controller.mkdir(parents=True, exist_ok=False)
    identity = {**identity, 'programme': 'full_13_person_13_fold_population_source', 'resource_scope': scope,
        'resource_owner': 'source_family_resource_owner', 'project_capacity_device': capacity['programme_capacity_device'],
        'actual_programme_device': 'cpu', 'programme_phase': asdict(phase), 'ready_gpu_tasks': 0,
        'programme_gpu_claim': 0, 'internal_source_workers': resources['source_workers'],
        'internal_fold_workers': resources['fold_workers'], 'independent_acceptance_issued': False,
        'whole_machine_utilization_supervision': 'external_launch_supervisor' if scope == 'local_development' else None}
    task = CloudTask('population-source-complete', 'research.population_source_cloud:execute_population_programme',
                     {'argv': list(argv), 'capacity': capacity}, str(output), phase)

    def acknowledge(outcome):
        if outcome.status != 'completed':
            return ()
        reference = outcome.value
        content = Path(reference['path']).read_bytes()
        if len(content) != reference['bytes'] or hashlib.sha256(content).hexdigest() != reference['sha256']:
            raise ValueError("the retired actor's actual source publication changed")
        record = json.loads(content)
        if (record['schema'] != 'complete_original_population_source_execution/1'
                or record['complete'] is not True or len(record['persons']) != 13 or len(record['folds']) != 13):
            raise ValueError('completion requires the actual complete thirteen-person/thirteen-fold publication')
        return ()

    with get_context('spawn').Manager() as manager:
        admission = HostMemoryAdmission(manager, capacity)
        executor = CloudExecutor(admission, max_workers=1, diagnostic_directory=controller, identity=identity)
        run = executor.run((task,), on_finished=acknowledge)
        receipt = {'schema': ('owned_cloud_population_source_programme/1' if scope == 'cloud'
                              else 'owned_local_population_source_programme/1'), 'identity': identity,
            'capacity': capacity, 'results': {key: asdict(value) for key, value in run.results.items()},
            'status': 'complete_pending_independent_acceptance' if run.status == 'completed' else 'failed',
            'diagnostics': run.diagnostics, 'observation_failures': run.observation_failures,
            'settled_resource_observation': admission.record(), 'source_requested': 13, 'fold_requested': 13,
            'ready_gpu_tasks': 0, 'programme_gpu_claim': 0, 'independent_acceptance_issued': False}
        filename = 'SOURCE_CLOUD_RECEIPT.json' if scope == 'cloud' else 'SOURCE_LOCAL_RECEIPT.json'
        atomic_write_json(controller / filename, receipt)
    if run.status != 'completed':
        raise RuntimeError('the owned source programme failed; original outputs and owner diagnostics retained')
    return run.results[task.task_id].value
